"""Quick screening of sleep variants on S8: the first 8 sessions of a LoCoMo conversation and the questions whose evidence lies in them.

    uv run python screen.py run baseline                               # 2B cards, diluted replay (the C4 recipe)
    uv run python screen.py run oracle --cards oracle_haiku            # one change: the cards
    uv run python screen.py run full_replay --replay full              # one change: the replay
    uv run python screen.py run hybrid_a --hybrid-top-k 2              # one change: top-2 cards in the prompt
    uv run python screen.py rag --cards oracle_haiku --top-k 1 2 4     # control: same cards in a RAG, bare model, no training
    uv run python screen.py compare baseline oracle                    # paired F1 difference on the same questions

Cards of an extraction variant live in cards_cache/<variant>/night_XX.json ([{q, a}]); the 2b variant is read
from the finished C4 run. Card generation and training are timed apart: generation in cards_cache/<variant>/timing.json
(when the cards were generated here), training per night in the run file.
"""

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import locomo
import locomo_run
from lm import LM
from locomo_run import LocoSleep
from sleep import GENERAL_QUESTIONS

HERE = Path(__file__).parent
CACHE = HERE / "cards_cache"
OUT = HERE / "runs" / "screen"
C4_STATE = HERE / "adapters" / "locomo_c4_sleep_replay_conv0_seed0" / "state.pkl"
SESSIONS = 8
RECALL_PER_NIGHT = 5
NOTES_PROMPT = "Notes about past conversations (with their dates):\n{notes}\n\n{question}"


def night_cards(variant: str, night: int, state: dict | None) -> list[tuple[str, str]]:
    path = CACHE / variant / f"night_{night:02d}.json"
    if not path.exists():
        if variant != "2b":
            raise SystemExit(f"{path} is missing: generate the cards of {variant!r} first (oracle_cards.py --variant {variant})")
        seen, cards = set(), []
        for q, a in state["log"][night - 1]["cards"]:
            if (q, a) not in seen:
                seen.add((q, a))
                cards.append({"q": q, "a": a})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cards, indent=1))
    return [(c["q"], c["a"]) for c in json.loads(path.read_text())]


class ScreenSleep(LocoSleep):
    """Nightly LoRA training on pre-written cards; the replay and the answering path are the knobs under test."""

    def __init__(self, lm: LM, seed: int, speakers, by_night: dict[int, list[tuple[str, str]]], replay: str, top_k: int):
        super().__init__(lm, seed, speakers)
        self.by_night, self.replay_mode, self.top_k = by_night, replay, top_k
        self.night = 0
        self.dates: dict[int, str] = {}
        self.train_s: list[float] = []

    def live_day(self, session: locomo.Session) -> None:
        self.night += 1
        self.dates[self.night] = locomo.date_only(session.date)

    def sleep(self) -> None:
        if not self.general:
            with self.lm.model.disable_adapter():
                answers = self.lm.complete_batch(GENERAL_QUESTIONS, system="Answer in one short sentence.", max_new_tokens=32)
            self.general = list(zip(GENERAL_QUESTIONS, answers))
        today = self.by_night[self.night]
        old = [c for n, cards in self.by_night.items() if n < self.night for c in cards]
        replayed = old if self.replay_mode == "full" else self.rng.sample(old, min(self.replay_size, len(old)))
        examples = list(today) + replayed + self.rng.sample(self.general, min(8, len(self.general)))
        t = time.time()
        loss = self.train(examples)
        self.train_s.append(round(time.time() - t))
        self.log.append({"today_cards": len(today), "examples": len(examples), "loss": loss, "cards": today})

    def attach_notes(self) -> None:
        """The cards written so far, searchable by embedding; top-k of them go in the prompt (next to the adapter, or alone)."""
        self.rag = locomo_run.LocoRag(self.lm, 1)
        self.rag.snippets = [f"({self.dates[n]}) {q} -> {a}" for n, cards in sorted(self.by_night.items()) for q, a in cards]
        self.rag.embeddings = self.rag.encoder.encode(self.rag.snippets, convert_to_tensor=True, normalize_embeddings=True)

    def answer(self, question: str):
        if not self.top_k:
            return super().answer(question)
        self.rag.top_k = self.top_k
        notes = "\n".join(self.rag.retrieve(question))
        return self.lm.chat([{"role": "user", "content": NOTES_PROMPT.format(notes=notes, question=question)}], max_new_tokens=24)


def card_recall(memory: ScreenSleep, seed: int = 0) -> dict:
    """F1 of the adapter alone on the questions of its own cards (a fixed draw per night): did the cards get stored?"""
    rng = random.Random(seed)
    scores = []
    for night, cards in sorted(memory.by_night.items()):
        for q, a in rng.sample(cards, min(RECALL_PER_NIGHT, len(cards))):
            reply = memory.lm.chat([{"role": "user", "content": q}], max_new_tokens=24)
            scores.append(locomo.f1_score(reply.text, a))
    return {"n": len(scores), "f1": 100 * sum(scores) / len(scores)}


def run(args) -> None:
    conv = locomo.load(0)
    with open(C4_STATE, "rb") as f:
        state = pickle.load(f)
    by_night = {n: night_cards(args.cards, n, state) for n in range(1, SESSIONS + 1)}
    questions = [q for q in conv.questions if q.answerable_after(SESSIONS)]
    speakers = tuple(sorted({s for sess in conv.sessions for s, _ in sess.turns}))[:2]
    memory = ScreenSleep(LM(system=locomo_run.ANSWER_SYSTEM), args.seed, speakers, by_night, args.replay, 0)
    for session in conv.sessions[:SESSIONS]:
        memory.live_day(session)
        memory.sleep()
        print(f"night {memory.night}: {len(memory.by_night[memory.night])} cards, trained in {memory.train_s[-1]} s", flush=True)
    recall = card_recall(memory)
    OUT.mkdir(parents=True, exist_ok=True)
    cards_total = sum(len(c) for c in by_night.values())

    def save(label: str, records: list[dict], **extra) -> None:
        f1 = 100 * sum(r["f1"] for r in records) / len(records)
        tokens = sum(r["prompt_tokens"] for r in records) / len(records)
        median = lambda key: sorted(r[key] for r in records)[len(records) // 2]  # noqa: E731
        (OUT / f"{label}_seed{args.seed}.json").write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()}, "questions": len(records), "f1": f1, "prompt_tokens": tokens,
            "median_prompt_tokens": median("prompt_tokens"), "median_prefill_s": median("prefill_s"),
            "card_recall": recall, "train_s": memory.train_s, "cards": cards_total, "records": records, **extra,
        }, indent=1))
        print(f"{label}: F1 {f1:.1f} over {len(records)} questions, {tokens:.0f} prompt tokens, card recall {recall['f1']:.1f}, "
              f"training {sum(memory.train_s) / 60:.1f} min", flush=True)

    save(args.name, locomo_run.evaluate(memory, questions))
    memory.lm.model.save_pretrained(HERE / "adapters" / f"screen_{args.name}_seed{args.seed}")
    memory.attach_notes()
    if args.hybrid_top_k:  # hybrid A: the adapter and the k closest cards
        memory.top_k = args.hybrid_top_k
        save(f"{args.name}_hybrid{args.hybrid_top_k}", locomo_run.evaluate(memory, questions))
    if args.rag_ref:  # the honest competitor: the same cards in a RAG, base model without adapter
        memory.top_k = args.rag_ref
        with memory.lm.model.disable_adapter():
            save(f"{args.name}_ragref{args.rag_ref}", locomo_run.evaluate(memory, questions))


def rag(args) -> None:
    """Control for hybrid A: the cards of a variant in a RAG over the bare model (no adapter, no training)."""
    conv = locomo.load(0)
    with open(C4_STATE, "rb") as f:
        state = pickle.load(f)
    by_night = {n: night_cards(args.cards, n, state) for n in range(1, SESSIONS + 1)}
    questions = [q for q in conv.questions if q.answerable_after(SESSIONS)]
    speakers = tuple(sorted({s for sess in conv.sessions for s, _ in sess.turns}))[:2]
    memory = ScreenSleep(LM(system=locomo_run.ANSWER_SYSTEM), args.seed, speakers, by_night, "diluted", 0)
    for session in conv.sessions[:SESSIONS]:
        memory.live_day(session)
    memory.attach_notes()
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.cards.removesuffix("_haiku")
    for k in args.top_k:
        memory.top_k = k
        records = locomo_run.evaluate(memory, questions)
        median = lambda key: sorted(r[key] for r in records)[len(records) // 2]  # noqa: E731
        f1 = 100 * sum(r["f1"] for r in records) / len(records)
        (OUT / f"{name}_ragref{k}_seed{args.seed}.json").write_text(json.dumps({
            "args": {a: str(v) for a, v in vars(args).items()} | {"top_k": str(k)}, "questions": len(records), "f1": f1,
            "prompt_tokens": sum(r["prompt_tokens"] for r in records) / len(records),
            "median_prompt_tokens": median("prompt_tokens"), "median_prefill_s": median("prefill_s"),
            "cards": sum(len(c) for c in by_night.values()), "records": records,
        }, indent=1))
        print(f"{name}_ragref{k}: F1 {f1:.1f} over {len(records)} questions, median {median('prompt_tokens')} tokens", flush=True)


def load(name: str) -> dict:
    return json.loads((OUT / f"{name}_seed0.json").read_text()) if "_seed" not in name else json.loads((OUT / f"{name}.json").read_text())


def compare(a: str, b: str) -> None:
    first, second = load(a), load(b)
    theirs = {r["question"]: r for r in second["records"]}
    pairs = [(r["f1"], theirs[r["question"]]["f1"], r["category"]) for r in first["records"] if r["question"] in theirs]
    diffs = [100 * (y - x) for x, y, _ in pairs]
    mean = sum(diffs) / len(diffs)
    se = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5 / len(diffs) ** 0.5
    rng = random.Random(0)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
    low, high = boot[249], boot[9749]
    print(f"{b} - {a}: {mean:+.1f} F1 (+-{2 * se:.1f}, two standard errors; paired bootstrap 95% [{low:+.1f}, {high:+.1f}]) on {len(diffs)} paired questions; "
          f"{a} {100 * sum(p[0] for p in pairs) / len(pairs):.1f}, {b} {100 * sum(p[1] for p in pairs) / len(pairs):.1f}")
    by_cat = defaultdict(list)
    for x, y, cat in pairs:
        by_cat[locomo.CATEGORIES[cat]].append(100 * (y - x))
    se_of = lambda v: (sum((d - sum(v) / len(v)) ** 2 for d in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5 if len(v) > 1 else float("nan")  # noqa: E731
    print("  " + "  ".join(f"{c} {sum(v) / len(v):+.1f} (+-{se_of(v):.1f} se, n={len(v)})" for c, v in sorted(by_cat.items())))
    recall = lambda d: f"{d['card_recall']['f1']:.1f}" if "card_recall" in d else "n/a (no adapter)"  # noqa: E731
    print(f"  card recall {recall(first)} -> {recall(second)}; prompt tokens {first['prompt_tokens']:.0f} -> {second['prompt_tokens']:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--cards", default="2b", help="extraction variant: a directory of cards_cache, or 2b for the cards of the C4 run")
    p.add_argument("--replay", choices=["diluted", "full"], default="diluted", help="diluted: 24 random old cards per night; full: all of them")
    p.add_argument("--hybrid-top-k", type=int, default=0, help="hybrid A: also answer with the adapter and the k closest cards in the prompt")
    p.add_argument("--rag-ref", type=int, default=10, help="also answer with a RAG top-k over the same cards and no adapter (0: skip)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("rag")
    p.add_argument("--cards", default="oracle_haiku")
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    args = parser.parse_args()
    {"run": run, "rag": rag}[args.command](args) if args.command != "compare" else compare(args.a, args.b)
