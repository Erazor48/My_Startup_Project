"""Diagnose why the sleep loses to RAG on LoCoMo, from the cards of a finished C4 run.

    uv run python diag_cards.py coverage   # (c) lexical card coverage, no model loaded
    uv run python diag_cards.py paired     # C4 against RAG on the same questions, no model loaded
"""

import argparse
import json
import pickle
import re
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


def coverage() -> None:
    conv = locomo.load(0)
    cards, state, night = load_cards()
    print(f"cards: {len(cards)} after night {night} ({len(set(cards))} distinct)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["coverage", "paired"])
    {"coverage": coverage, "paired": paired}[parser.parse_args().step]()
