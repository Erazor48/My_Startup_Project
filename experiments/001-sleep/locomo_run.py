"""LoCoMo with the sleep loop: each session is a day, then a night. Questions are asked after the last night.

    uv run python locomo_run.py c4_sleep_replay --conv 0
    uv run python locomo_run.py c2_rag --conv 0 --top-k 10
"""

import argparse
import json
import pickle
import random
import re
import time
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import locomo
from lm import LM, peak_rss_mb, reset_peak_rss
from locomo import Question, Session
from memory import NoMemory, Rag
from sleep import STOPWORDS, Sleep, SleepIntegrate, SleepReplay, parse_cards

HERE = Path(__file__).parent
ANSWER_SYSTEM = (
    "You answer questions about the life of two people from their past conversations, which you do not have in "
    "front of you. Answer with a short phrase (a few words), never a full sentence. If unsure, give your best guess."
)
DREAM_SYSTEM = "You are a memory system. Follow the format of the examples exactly."
DREAM_PROMPT = """Turn what a speaker said into flashcards for a memory. The questions will be asked later by someone who did not hear the conversation: they use the people's names, never "he" or "she". Answers are short (a few words) and repeat the name when the answer is a person. Turn relative dates ("yesterday", "last month") into calendar dates using the date of the conversation. If nothing is worth remembering, write NOTHING. When something happened, always write one card asking when, answered with a calendar date.

Date of conversation: 3 June 2022
Before: Marta: "How was your weekend?"
Said by Leo: "Great! I finally repaired my bicycle yesterday and rode it to the lake."
Flashcards:
Q: What did Leo repair?
A: His bicycle
Q: When did Leo repair his bicycle?
A: 2 June 2022
Q: Where did Leo ride his bicycle?
A: To the lake

Date of conversation: 15 March 2021
Before: Leo: "Do you see your family often?"
Said by Marta: "My brother Paul moved to Oslo last month, so not really."
Flashcards:
Q: Who is Marta's brother?
A: Paul
Q: Where does Paul live?
A: Oslo
Q: When did Paul move to Oslo?
A: February 2021

Date of conversation: 9 July 2020
Before: Marta: "See you next week!"
Said by Leo: "Thanks, that sounds fun!"
Flashcards:
NOTHING

Date of conversation: {date}
Before: {context}
Said by {speaker}: "{text}"
Flashcards:
"""
INTEGRATE_PROMPT = """Two memories mention the same thing. Write flashcards about what follows from combining them. The questions will be asked later by someone who did not hear the conversations: they use names. Answers are short.

Old: [15 March 2021] Marta: My brother Paul moved to Oslo.
New: [2 May 2021] Leo: Paul found a job at Volvo.
Flashcards:
Q: Where does Marta's brother work?
A: Volvo
Q: In which city does Volvo employ Marta's brother?
A: Oslo

Old: {old}
New: {new}
Flashcards:
"""
MAX_CONTEXT = 16000  # tokens of history kept by the full-context condition
CURVE_QUESTIONS = 30  # questions per intermediate checkpoint (the full-context condition is slow)
DREAM_CHUNK = 12
MAX_PAIRS = 24
PROPER_NOUN = re.compile(r"(?<![.!?]\s)(?<!^)(?<!: )\b[A-Z][a-z]{2,}\b")


def first_block(text: str) -> str:
    """Keep the first answer only: the model restates the prompt's last lines, and may go on writing new examples."""
    if "Flashcards:" in text:
        text = text.split("Flashcards:", 1)[1]
    return re.split(r"\n\s*(?:Date of conversation|Said by|Old|New|Flashcards):", text)[0]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()) if w not in STOPWORDS and len(w) > 2}


def grounded(cards, *sources, ignore=frozenset()):
    """Keep cards whose answer brings a word of the source memories that the question lacks (numbers included)."""
    vocab = set().union(*(words(s) for s in sources)) - ignore
    return [(q, a) for q, a in cards if (words(a) - words(q)) & vocab]


def proper_nouns(text: str, speakers: set[str]) -> set[str]:
    return set(PROPER_NOUN.findall(text)) - speakers


class LocoNone(NoMemory):
    name = "c0_none"

    def live_day(self, session: Session) -> None:
        pass

    def answer(self, question: str):
        return self.lm.chat([{"role": "user", "content": question}], max_new_tokens=24)


class LocoRag(Rag):
    """C2 on LoCoMo: every turn is a snippet carrying its date; the top-k closest ones go in the prompt."""

    name = "c2_rag"

    def __init__(self, lm: LM, top_k: int = 4):
        super().__init__(lm)
        self.top_k = top_k

    def live_day(self, session: Session) -> None:
        date = locomo.date_only(session.date)
        new = [f"({date}) {speaker}: {text}" for speaker, text in session.turns]
        emb = self.encoder.encode(new, convert_to_tensor=True, normalize_embeddings=True)
        self.snippets += new
        self.embeddings = emb if self.embeddings is None else torch.cat([self.embeddings, emb])

    def answer(self, question: str):
        memories = "\n".join(self.retrieve(question))
        prompt = f"Things said in past conversations (with their dates):\n{memories}\n\n{question}"
        return self.lm.chat([{"role": "user", "content": prompt}], max_new_tokens=24)


class LocoFull(NoMemory):
    """C1 on LoCoMo: the whole history in the prompt, cut to the most recent MAX_CONTEXT tokens when it does not fit."""

    name = "c1_full_context"

    def __init__(self, lm: LM):
        super().__init__(lm)
        self.lines: list[str] = []
        self.truncated = False

    def live_day(self, session: Session) -> None:
        date = locomo.date_only(session.date)
        self.lines += [f"({date}) {speaker}: {text}" for speaker, text in session.turns]

    def answer(self, question: str):
        tok = self.lm.tokenizer
        history = "\n".join(self.lines)
        ids = tok(history, add_special_tokens=False).input_ids
        if len(ids) > MAX_CONTEXT:
            self.truncated = True
            history = tok.decode(ids[-MAX_CONTEXT:])
        prompt = f"Past conversations (with their dates):\n{history}\n\n{question}"
        return self.lm.chat([{"role": "user", "content": prompt}], max_new_tokens=24)


class LocoSleep(SleepReplay):
    """C4 on LoCoMo: dream (cards with calendar dates) -> replay -> LoRA. Cards are short question/answer pairs."""

    name = "c4_sleep_replay"
    chunk = DREAM_CHUNK

    def __init__(self, lm: LM, seed: int = 0, speakers: tuple[str, str] = ("", "")):
        super().__init__(lm, seed)
        self.speakers = set(speakers)
        self.meta: dict[str, tuple[str, str, str, str]] = {}  # key -> (date, context, speaker, text)

    def live_day(self, session: Session) -> None:
        date = locomo.date_only(session.date)
        previous = ""
        for speaker, text in session.turns:
            key = f"[{date}] {speaker}: {text}"
            self.meta[key] = (date, previous, speaker, text)
            self.buffer.append(key)
            previous = f"{speaker}: {text}"

    def _complete(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        with self.lm.model.disable_adapter():
            out = []
            for i in range(0, len(prompts), self.chunk):
                out += self.lm.complete_batch(prompts[i:i + self.chunk], system=DREAM_SYSTEM, max_new_tokens=max_new_tokens)
            return out

    def dream(self, utterances: list[str]) -> list[tuple[str, str]]:
        prompts = []
        for key in utterances:
            date, context, speaker, text = self.meta[key]
            prompts.append(DREAM_PROMPT.format(date=date, context=context or "(start of the conversation)", speaker=speaker, text=text))
        cards, seen = [], set()
        for key, out in zip(utterances, self._complete(prompts, 160)):
            context = self.meta[key][1]
            parsed = [c for c in grounded(parse_cards(first_block(out)), key, context) if c not in seen]  # the model also rewrites the turn before
            seen.update(parsed)
            self.cards[key] = parsed
            cards += parsed
        return cards

    def integrate_with_past(self, utterances: list[str]) -> list[tuple[str, str]]:
        pairs = [
            (p, u) for u in utterances for p in self.past
            if proper_nouns(self.meta[u][3], self.speakers) & proper_nouns(self.meta[p][3], self.speakers)
        ]
        if len(pairs) > MAX_PAIRS:
            pairs = self.rng.sample(pairs, MAX_PAIRS)
        if not pairs:
            return []
        outputs = self._complete([INTEGRATE_PROMPT.format(old=p, new=u) for p, u in pairs], 160)
        cards = []
        for (old, new), out in zip(pairs, outputs):
            shared = {w.lower() for w in proper_nouns(self.meta[old][3], self.speakers) & proper_nouns(self.meta[new][3], self.speakers)}
            parsed = grounded(parse_cards(first_block(out)), old, new, ignore=shared)
            self.cards[new] = self.cards.get(new, []) + parsed
            cards += parsed
        return cards

    def answer(self, question: str):
        return self.lm.chat([{"role": "user", "content": question}], max_new_tokens=24)


class LocoSleepIntegrate(LocoSleep):
    name = "c5_sleep_integrate"
    integrate = True


class RawLora(LocoSleep):
    """The failure published by Zedong Peng (2026): LoRA trained on the raw text of the conversation, no cards."""

    name = "c3_lora_raw"
    block = 512
    batch_size = 4

    def sleep(self) -> None:
        text = "\n".join(self.buffer)
        date = self.meta[self.buffer[0]][0]
        loss = self.train_text(f"Conversation of {date}.\n" + "\n".join(f"{self.meta[k][2]}: {self.meta[k][3]}" for k in self.buffer))
        self.log.append({"today_cards": 0, "examples": len(self.buffer), "loss": loss, "cards": []})
        self.past += self.buffer
        self.buffer = []
        del text

    def train_text(self, text: str) -> float:
        tok = self.lm.tokenizer
        ids = tok(text, add_special_tokens=False).input_ids
        blocks = [ids[i:i + self.block] for i in range(0, len(ids), self.block)]
        model = self.lm.model
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        losses = []
        for _ in range(self.epochs):
            self.rng.shuffle(blocks)
            for i in range(0, len(blocks), self.batch_size):
                batch = blocks[i:i + self.batch_size]
                width = max(map(len, batch))
                input_ids = torch.tensor([b + [tok.pad_token_id] * (width - len(b)) for b in batch])
                labels = torch.tensor([b + [-100] * (width - len(b)) for b in batch])
                mask = torch.tensor([[1] * len(b) + [0] * (width - len(b)) for b in batch])
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
        model.eval()
        return sum(losses[-5:]) / min(5, len(losses))


SLEEPERS = {c.name: c for c in (RawLora, LocoSleep, LocoSleepIntegrate)}
CONDITIONS = ["c0_none", "c1_full_context", "c2_rag", *SLEEPERS]


def build(condition: str, lm: LM, seed: int, speakers: tuple[str, str], top_k: int):
    if condition == "c0_none":
        return LocoNone(lm)
    if condition == "c1_full_context":
        return LocoFull(lm)
    if condition == "c2_rag":
        return LocoRag(lm, top_k)
    return SLEEPERS[condition](lm, seed, speakers)


def save_state(memory: Sleep, directory: Path, night: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(memory.lm.model), directory / "adapter.pt")
    state = {k: getattr(memory, k) for k in ("past", "cards", "general", "log", "meta")}
    state.update(night=night, rng=memory.rng.getstate())
    with open(directory / "state.pkl", "wb") as f:
        pickle.dump(state, f)


def load_state(memory: Sleep, directory: Path) -> int:
    set_peft_model_state_dict(memory.lm.model, torch.load(directory / "adapter.pt"))
    with open(directory / "state.pkl", "rb") as f:
        state = pickle.load(f)
    for k in ("past", "cards", "general", "log", "meta"):
        setattr(memory, k, state[k])
    memory.rng.setstate(state["rng"])
    return state["night"]


def evaluate(memory, questions: list[Question]) -> list[dict]:
    records = []
    for i, q in enumerate(questions):
        t = time.time()
        reply = memory.answer(q.question)
        records.append({
            "category": q.category, "question": q.question, "expected": q.answer, "answer": reply.text,
            "f1": locomo.question_f1(q, reply.text), "prompt_tokens": reply.prompt_tokens, "answer_s": round(time.time() - t, 2),
            "prefill_s": reply.prefill_s, "peak_rss_mb": reply.peak_rss_mb,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(questions)} questions, F1 so far {sum(r['f1'] for r in records) / len(records):.3f}", flush=True)
    return records


def pick(questions: list[Question], n_sessions: int, final: bool, cap: int | None) -> list[Question]:
    """Questions whose evidence lies in the first n sessions (all of them at the end), at most `cap` of them."""
    pool = questions if final else [q for q in questions if q.answerable_after(n_sessions)]
    if cap and len(pool) > cap:
        keep = set(random.Random(0).sample(range(len(pool)), cap))
        pool = [q for i, q in enumerate(pool) if i in keep]
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--nights", type=int, help="only live this many sessions (smoke test)")
    parser.add_argument("--eval-at", type=int, nargs="*", default=[1, 5, 10, 20], help="intermediate checkpoints (sessions)")
    parser.add_argument("--curve-questions", type=int, default=CURVE_QUESTIONS, help="questions per intermediate checkpoint")
    parser.add_argument("--final-questions", type=int, help="cap on the final questions (default: all, 30 for full context)")
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "locomo")
    args = parser.parse_args()

    conv = locomo.load(args.conv)
    sessions = conv.sessions[:args.nights]
    last = len(sessions)
    final_cap = args.final_questions or (CURVE_QUESTIONS if args.condition == "c1_full_context" else None)
    checkpoints = sorted({n for n in args.eval_at if n < last} | {last})
    name = f"{args.condition}{'_k' + str(args.top_k) if args.condition == 'c2_rag' else ''}_conv{args.conv}"
    speakers = tuple(sorted({s for sess in conv.sessions for s, _ in sess.turns}))[:2]
    memory = build(args.condition, LM(system=ANSWER_SYSTEM), args.seed, speakers, args.top_k)

    nights, curve, start = [], {}, 0
    state_dir = HERE / "adapters" / f"locomo_{name}_seed{args.seed}"
    if args.condition in SLEEPERS and (state_dir / "state.pkl").exists():
        start = load_state(memory, state_dir)
        nights = json.loads((state_dir / "nights.json").read_text())
        curve = json.loads((state_dir / "curve.json").read_text())
        print(f"resumed after night {start}", flush=True)

    for night, session in enumerate(sessions, 1):
        memory.live_day(session)
        if night > start:
            reset_peak_rss()
            t = time.time()
            memory.sleep()
            entry = {"night": night, "date": session.date, "turns": len(session.turns), "sleep_s": round(time.time() - t), "peak_rss_mb": round(peak_rss_mb())}
            if hasattr(memory, "log") and memory.log:
                entry.update(cards=memory.log[-1]["today_cards"], examples=memory.log[-1]["examples"], loss=memory.log[-1]["loss"])
            nights.append(entry)
            print(f"night {night}/{last}: {entry}", flush=True)
        if night in checkpoints and str(night) not in curve:
            final = night == last
            qs = pick(conv.questions, night, final, final_cap if final else args.curve_questions)
            print(f"checkpoint {night}: asking {len(qs)} questions", flush=True)
            curve[str(night)] = evaluate(memory, qs)
        if night > start and args.condition in SLEEPERS:
            save_state(memory, state_dir, night)
            (state_dir / "nights.json").write_text(json.dumps(nights))
            (state_dir / "curve.json").write_text(json.dumps(curve))

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{name}.json"
    out.write_text(json.dumps({
        "args": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "n_sessions": last},
        "sample_id": conv.sample_id, "nights": nights, "curve": curve, "records": curve[str(last)],
        "truncated": getattr(memory, "truncated", False),
        "cards": {k: v for k, v in getattr(memory, "cards", {}).items() if v},
    }, indent=1))
    print(f"saved {out}")
    if args.condition in SLEEPERS:
        memory.lm.model.save_pretrained(HERE / "adapters" / f"locomo_{name}")


if __name__ == "__main__":
    main()
