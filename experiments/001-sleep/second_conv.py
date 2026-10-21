"""Reader side of the second-conversation test: the 2B reader, no adapter, same settings as S8, on a whole conversation.

    uv run python second_conv.py --conv 2 --sessions 32 --cards local_qwen35_4b_conv41

Conditions on the questions answerable in the sessions lived: C0 (no memory), RAG over raw turns (k=2, 10), RAG over the
cards of the local writer (k=2, 10). Intermediate checkpoints (sessions 1, 10, 20) ask a fixed draw of --curve-questions
questions to give tokens and prefill along the conversation; their F1 is not a result. Files go to runs/screen/c<conv>s<sessions>_*,
readable by `screen.py compare`.
"""

import argparse
import json
import random

import torch

import locomo
import locomo_run
from lm import LM
from locomo_run import LocoNone, LocoRag
from screen import NOTES_PROMPT, OUT, night_cards


class CardRag(LocoRag):
    """RAG over the cards of the nights lived so far, same snippet format and prompt as screen.py's attach_notes."""

    def __init__(self, lm: LM, by_night: dict[int, list[tuple[str, str]]]):
        super().__init__(lm, 2)
        self.by_night, self.night = by_night, 0

    def live_day(self, session: locomo.Session) -> None:
        self.night += 1
        date = locomo.date_only(session.date)
        new = [f"({date}) {q} -> {a}" for q, a in self.by_night[self.night]]
        if not new:
            return
        emb = self.encoder.encode(new, convert_to_tensor=True, normalize_embeddings=True)
        self.snippets += new
        self.embeddings = emb if self.embeddings is None else torch.cat([self.embeddings, emb])

    def answer(self, question: str):
        notes = "\n".join(self.retrieve(question))
        return self.lm.chat([{"role": "user", "content": NOTES_PROMPT.format(notes=notes, question=question)}], max_new_tokens=24)


def stats(records: list[dict]) -> dict:
    median = lambda key: sorted(r[key] for r in records)[len(records) // 2]  # noqa: E731
    return {"questions": len(records), "f1": 100 * sum(r["f1"] for r in records) / len(records),
            "prompt_tokens": sum(r["prompt_tokens"] for r in records) / len(records),
            "median_prompt_tokens": median("prompt_tokens"), "median_prefill_s": median("prefill_s")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--curve-at", type=int, nargs="*", default=[1, 10, 20])
    parser.add_argument("--curve-questions", type=int, default=30)
    args = parser.parse_args()

    conv = locomo.load(args.conv)
    prefix = f"c{args.conv}s{args.sessions}_"
    by_night = {n: night_cards(args.cards, n, None) for n in range(1, args.sessions + 1)}
    sessions = conv.sessions[:args.sessions]
    lm = LM(system=locomo_run.ANSWER_SYSTEM)
    # (label, memory, top_k): raw and card memories are filled session by session so that one pass serves every checkpoint
    c0 = LocoNone(lm)
    raw = LocoRag(lm, 2)
    cards = CardRag(lm, by_night)
    conditions = [("c0_none", c0, None), ("raw_ragref2", raw, 2), ("raw_ragref10", raw, 10),
                  ("local_ragref2", cards, 2), ("local_ragref10", cards, 10)]
    OUT.mkdir(parents=True, exist_ok=True)
    curve = {label: {} for label, _, _ in conditions}
    for night, session in enumerate(sessions, 1):
        raw.live_day(session)
        cards.live_day(session)
        if night in args.curve_at:
            pool = [q for q in conv.questions if q.answerable_after(night)]
            if len(pool) > args.curve_questions:
                keep = set(random.Random(0).sample(range(len(pool)), args.curve_questions))
                pool = [q for i, q in enumerate(pool) if i in keep]
            for label, memory, k in conditions:
                if k:
                    memory.top_k = k
                curve[label][str(night)] = stats(locomo_run.evaluate(memory, pool))
                print(f"checkpoint {night} {label}: {curve[label][str(night)]}", flush=True)
    questions = [q for q in conv.questions if q.answerable_after(args.sessions)]
    print(f"{len(questions)} questions answerable in {args.sessions} sessions", flush=True)
    for label, memory, k in conditions:
        if k:
            memory.top_k = k
        records = locomo_run.evaluate(memory, questions)
        final = stats(records)
        curve[label][str(args.sessions)] = {kk: vv for kk, vv in final.items()}
        (OUT / f"{prefix}{label}_seed0.json").write_text(json.dumps({
            "args": {"conv": args.conv, "sessions": args.sessions, "cards": args.cards, "top_k": k}, **final,
            "cards": sum(len(c) for c in by_night.values()), "curve": curve[label], "records": records}, indent=1))
        print(f"{label}: F1 {final['f1']:.1f} over {final['questions']} questions, median {final['median_prompt_tokens']} tokens, "
              f"median prefill {final['median_prefill_s']} s", flush=True)


if __name__ == "__main__":
    main()
