"""Diagnose why the sleep loses to RAG on LoCoMo, from the cards of a finished C4 run.

    uv run python diag_cards.py coverage   # (c) lexical card coverage, no model loaded
    uv run python diag_cards.py paired     # C4 against RAG on the same questions, no model loaded
"""

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import locomo
from sleep import STOPWORDS

HERE = Path(__file__).parent
RUNS = HERE / "runs" / "locomo"
STATE = HERE / "adapters" / "locomo_c4_sleep_replay_conv0_seed0" / "state.pkl"
COVERED_RECALL = 0.8  # share of the gold answer's content stems the card's answer must hold to count as covering it


def stems(text: str) -> set[str]:
    return {locomo.STEM.stem(w) for w in locomo.normalize(text).split() if w not in STOPWORDS and len(w) > 1}


def load_cards(path: Path = STATE) -> tuple[list[tuple[str, str]], dict, int]:
    with open(path, "rb") as f:
        state = pickle.load(f)
    cards = [c for per_turn in state["cards"].values() for c in per_turn]
    return cards, state, state["night"]


def covering_card(question: locomo.Question, cards: list[tuple[str, str]]) -> tuple[str, str] | None:
    """A card covers a question if its text holds enough of the gold answer and shares a topic word with the question."""
    gold = stems(question.answer.split(";")[0] if question.category == 3 else question.answer)
    topic = stems(question.question) - gold
    if not gold:
        return None
    best, best_score = None, 0.0
    for q, a in cards:
        card = stems(a)
        recall = len(gold & card) / len(gold)
        if recall >= COVERED_RECALL and topic & stems(q) and recall > best_score:
            best, best_score = (q, a), recall
    return best


def load_records(name: str) -> dict[str, dict]:
    data = json.loads((RUNS / f"{name}.json").read_text())
    return {r["question"]: r for r in data["records"]}


def mean(values: list[float]) -> float:
    return 100 * sum(values) / len(values) if values else float("nan")


def coverage(cards_path: Path | None = None) -> None:
    conv = locomo.load(0)
    cards, state, night = load_cards()
    if cards_path:
        cards = [(c["q"], c["a"]) for c in json.loads(cards_path.read_text())]
    print(f"cards: {len(cards)} ({len(set(cards))} distinct)" + ("" if cards_path else f" after night {night}"))
    covered = {q.question: covering_card(q, cards) for q in conv.questions}
    c4 = load_records("c4_sleep_replay_conv0") if (RUNS / "c4_sleep_replay_conv0.json").exists() else {}
    rows = defaultdict(lambda: {"covered": [], "uncovered": []})
    for q in conv.questions:
        side = "covered" if covered[q.question] else "uncovered"
        for key in ("all", locomo.CATEGORIES[q.category]):
            rows[key][side].append(c4[q.question]["f1"] if q.question in c4 else None)
    print(f"{'category':12} {'questions':>9} {'covered':>8} {'F1 cov':>7} {'F1 unc':>7}")
    for key, side in rows.items():
        n_cov, n_unc = len(side["covered"]), len(side["uncovered"])
        f_cov = mean([v for v in side["covered"] if v is not None])
        f_unc = mean([v for v in side["uncovered"] if v is not None])
        print(f"{key:12} {n_cov + n_unc:>9} {n_cov:>8} {f_cov:>7.1f} {f_unc:>7.1f}")


def paired() -> None:
    c4 = load_records("c4_sleep_replay_conv0")
    for rag_name in ("c2_rag_k4_conv0", "c2_rag_k10_conv0", "c0_none_conv0"):
        other = load_records(rag_name)
        shared = [q for q in c4 if q in other]
        print(f"C4 vs {rag_name}: {len(shared)} shared questions, "
              f"C4 {mean([c4[q]['f1'] for q in shared]):.1f}, {rag_name} {mean([other[q]['f1'] for q in shared]):.1f}")
    print("checkpoints (C4 against RAG k=10 on the questions both were asked):")
    c4_run = json.loads((RUNS / "c4_sleep_replay_conv0.json").read_text())
    rag_run = json.loads((RUNS / "c2_rag_k10_conv0.json").read_text())
    for point, records in c4_run["curve"].items():
        theirs = {r["question"]: r for r in rag_run["curve"].get(point, [])}
        shared = [r for r in records if r["question"] in theirs]
        print(f"  after {point:>2} sessions: {len(records)} C4 questions, {len(shared)} shared, "
              f"C4 {mean([r['f1'] for r in shared]):.1f}, RAG {mean([theirs[r['question']]['f1'] for r in shared]):.1f}")


def dated_cards(path: Path | None = None) -> list[dict]:
    """Distinct cards with the night they were written: from the C4 state, or from a JSON list of {night, q, a}."""
    if path:
        return json.loads(path.read_text())
    conv = locomo.load(0)
    _, state, _ = load_cards()
    seen, out = set(), []
    for night, entry in enumerate(state["log"], 1):
        for q, a in entry["cards"]:
            if (q, a) not in seen:
                seen.add((q, a))
                out.append({"night": night, "date": locomo.date_only(conv.sessions[night - 1].date), "q": q, "a": a})
    return out


def summary(name: str, records: list[dict]) -> None:
    by_cat = defaultdict(list)
    for r in records:
        by_cat[locomo.CATEGORIES[r["category"]]].append(r["f1"])
    cats = "  ".join(f"{c} {mean(v):.1f} (n={len(v)})" for c, v in sorted(by_cat.items()))
    tokens = sum(r["prompt_tokens"] for r in records) / len(records)
    print(f"{name}: F1 {mean([r['f1'] for r in records]):.1f} over {len(records)} questions, {tokens:.0f} prompt tokens | {cats}")


def save_records(name: str, records: list[dict], extra: dict) -> None:
    (RUNS / f"{name}.json").write_text(json.dumps({"records": records, **extra}, indent=1))
    summary(name, records)


def card_rag(top_k: int, cards_path: Path | None, tag: str) -> None:
    """(a) RAG over the cards, formatted '(date) Q -> A', instead of over the raw turns."""
    import locomo_run
    from lm import LM

    conv = locomo.load(0)
    cards = dated_cards(cards_path)
    rag = locomo_run.LocoRag(LM(system=locomo_run.ANSWER_SYSTEM), top_k)
    rag.snippets = [f"({c['date']}) {c['q']} -> {c['a']}" for c in cards]
    rag.embeddings = rag.encoder.encode(rag.snippets, convert_to_tensor=True, normalize_embeddings=True)

    def answer(question: str):
        memories = "\n".join(rag.retrieve(question))
        prompt = f"Notes about past conversations (with their dates):\n{memories}\n\n{question}"
        return rag.lm.chat([{"role": "user", "content": prompt}], max_new_tokens=24)

    rag.answer = answer
    t = time.time()
    records = locomo_run.evaluate(rag, conv.questions)
    save_records(f"diag_a_{tag}_rag_k{top_k}_conv0", records, {"cards": len(cards), "eval_s": round(time.time() - t)})


def recall_probes(memory, cards: list[dict], seed: int = 0) -> dict:
    """(b) Ask the card's own question to the adapter: 100 cards drawn evenly over the nights, plus the 20 hand-written paraphrases."""
    import locomo_run

    rng = random.Random(seed)
    nights = sorted({c["night"] for c in cards})
    extra = set(nights[::4][:100 - 5 * len(nights)]) if len(nights) < 20 else set()
    drawn = []
    for night in nights:
        pool = [c for c in cards if c["night"] == night]
        drawn += rng.sample(pool, min(len(pool), 5 + (night in extra)))
    paraphrases = json.loads((HERE / "diag_paraphrases.json").read_text())
    out = {"cards": [], "paraphrases": []}
    for c in drawn:
        reply = memory.answer(c["q"])
        out["cards"].append({"night": c["night"], "q": c["q"], "gold": c["a"], "answer": reply.text, "f1": locomo.f1_score(reply.text, c["a"])})
    for p in paraphrases:
        reply = memory.answer(p["paraphrase"])
        out["paraphrases"].append({"night": p["night"], "q": p["paraphrase"], "gold": p["answer"], "answer": reply.text, "f1": locomo.f1_score(reply.text, p["answer"])})
    return out


def print_recall(label: str, probes: dict) -> None:
    groups = {"nights 1-5": (1, 5), "nights 6-10": (6, 10), "nights 11-15": (11, 15), "nights 16-19": (16, 19)}
    parts = []
    for name, (lo, hi) in groups.items():
        sel = [r["f1"] for r in probes["cards"] if lo <= r["night"] <= hi]
        parts.append(f"{name} {mean(sel):.1f} (n={len(sel)}, >=0.5: {100 * sum(v >= 0.5 for v in sel) / max(len(sel), 1):.0f}%)")
    para = [r["f1"] for r in probes["paraphrases"]]
    own = [r["f1"] for r in probes["cards"]]
    print(f"{label}: cards {mean(own):.1f} | " + " | ".join(parts) + f" | paraphrases {mean(para):.1f} (n={len(para)})")


def recall(state_dir: Path) -> None:
    """(b) Recall of the cards by the final C4 adapter, against the base model without adapter."""
    import locomo_run
    from lm import LM

    conv = locomo.load(0)
    speakers = tuple(sorted({s for sess in conv.sessions for s, _ in sess.turns}))[:2]
    memory = locomo_run.LocoSleep(LM(system=locomo_run.ANSWER_SYSTEM), 0, speakers)
    locomo_run.load_state(memory, state_dir)
    cards = dated_cards()
    t = time.time()
    with_adapter = recall_probes(memory, cards)
    with memory.lm.model.disable_adapter():
        base = recall_probes(memory, cards)
    print_recall("adapter", with_adapter)
    print_recall("base   ", base)
    (RUNS / "diag_b_recall_conv0.json").write_text(json.dumps({"adapter": with_adapter, "base": base, "s": round(time.time() - t)}, indent=1))


def one_pass(cards_path: Path | None, tag: str) -> None:
    """(d) Ceiling of the schedule: one training on every card at once instead of night after night."""
    import locomo_run
    from lm import LM

    conv = locomo.load(0)
    speakers = tuple(sorted({s for sess in conv.sessions for s, _ in sess.turns}))[:2]
    memory = locomo_run.LocoSleep(LM(system=locomo_run.ANSWER_SYSTEM), 0, speakers)
    _, state, _ = load_cards()
    cards = dated_cards(cards_path)
    examples = [(c["q"], c["a"]) for c in cards] + list(state["general"])
    t = time.time()
    loss = memory.train(examples)
    train_s = round(time.time() - t)
    print(f"trained on {len(examples)} examples in {train_s} s, loss {loss:.3f}", flush=True)
    records = locomo_run.evaluate(memory, conv.questions)
    save_records(f"diag_d_onepass_{tag}_conv0", records, {"examples": len(examples), "loss": loss, "train_s": train_s})
    print_recall("one-pass recall", recall_probes(memory, dated_cards()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["coverage", "paired", "rag", "recall", "onepass"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cards", type=Path, help="JSON list of {night, date, q, a} instead of the cards of the C4 state")
    parser.add_argument("--tag", default="2b", help="label of the card source in the output names")
    args = parser.parse_args()
    {
        "coverage": lambda: coverage(args.cards), "paired": paired,
        "rag": lambda: card_rag(args.top_k, args.cards, args.tag),
        "recall": lambda: recall(STATE.parent),
        "onepass": lambda: one_pass(args.cards, args.tag),
    }[args.step]()
