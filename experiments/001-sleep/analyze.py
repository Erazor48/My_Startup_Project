"""Summarize runs/*.json into a table (stdout) and recall curves (results/recall.png).

    uv run python analyze.py
"""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from evaluate import score
from world import make_world

HERE = Path(__file__).parent
KINDS = ("direct", "indirect", "chain")
# Validated categorical palette (dataviz skill), fixed order, plus a marker shape per condition.
STYLE = {
    "c0_none": ("#2a78d6", "o", "C0 none"),
    "c1_full_context": ("#eb6834", "s", "C1 full context"),
    "c2_rag": ("#1baf7a", "D", "C2 RAG"),
    "c3_sleep_naive": ("#eda100", "v", "C3 sleep"),
    "c4_sleep_replay": ("#e87ba4", "^", "C4 sleep+replay"),
    "c5_sleep_integrate": ("#008300", "P", "C5 sleep+integration"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BASE_MODEL = {"c0_none", "c1_full_context", "c2_rag"}  # conditions that never change the weights


def recall(records: list[dict]) -> float | None:
    return mean(r["result"] == "correct" for r in records) if records else None


def load() -> dict[str, list[dict]]:
    """Load every run and re-score its answers, so a scoring fix never requires re-running the model."""
    runs = {}
    for path in sorted((HERE / "runs").glob("*.json")):
        data = json.loads(path.read_text())
        world = make_world(n_days=data["args"]["days"], seed=data["args"]["seed"])
        probes = {}
        for day in {r["day"] for r in data["records"]}:
            probes.update({(day, p.kind, p.slot): p for p in world.probes(day)})
        for r in data["records"]:
            r["result"] = score(r["answer"], probes[r["day"], r["kind"], r["slot"]])
        cond = path.stem.rsplit("_seed", 1)[0]
        runs.setdefault(cond, []).extend(data["records"])
    return {c: runs[c] for c in STYLE if c in runs}


def arc_scores(folder: str = "arc") -> dict[str, float]:
    """Mean ARC-Easy accuracy per condition over seeds, and 'base' for the model before any sleep."""
    scores = {}
    for path in (HERE / "runs" / folder).glob("*.json"):
        scores.setdefault(path.stem.rsplit("_seed", 1)[0], []).append(json.loads(path.read_text())["accuracy"])
    return {name: mean(accs) for name, accs in scores.items()}


def table(runs: dict[str, list[dict]]) -> None:
    arc, cloze = arc_scores(), arc_scores("arc_cloze")
    print(f"{'condition':22s} {'direct':>7s} {'indirect':>9s} {'chain':>6s} {'stale':>6s} {'old facts':>10s} {'tokens':>7s} {'ARC':>6s} {'cloze':>6s}")
    for cond, recs in runs.items():
        last = max(r["day"] for r in recs)
        final = [r for r in recs if r["day"] == last]
        cells = [recall([r for r in final if r["kind"] == k]) for k in KINDS]
        updated = [r for r in final if r["updated"]]
        stale = mean(r["result"] == "stale" for r in updated) if updated else None
        old = recall([r for r in final if r["kind"] != "chain" and r["fact_day"] <= 3])
        tokens = mean(r["prompt_tokens"] for r in final)
        key = "base" if cond in BASE_MODEL else cond
        general, general_cloze = arc.get(key), cloze.get(key)
        fmt = lambda v: "-" if v is None else f"{v:.0%}"
        print(f"{cond:22s} {fmt(cells[0]):>7s} {fmt(cells[1]):>9s} {fmt(cells[2]):>6s} {fmt(stale):>6s} {fmt(old):>10s} {tokens:7.0f} {'-' if general is None else f'{general:.1%}':>6s} {'-' if general_cloze is None else f'{general_cloze:.1%}':>6s}")
    print("\nAll columns at the last evaluated day. stale = updated facts answered with the old value.")
    print("old facts = direct+indirect recall on facts from days 1-3 (forgetting). tokens = mean prompt tokens.")
    print("ARC = ARC-Easy accuracy (200 questions, answer letter) after the last night; conditions without sleep use the base model.")
    print("cloze = same questions scored by the likelihood of each option's text (arc_cloze.py), no letters.")


def plot(runs: dict[str, list[dict]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for cond, recs in runs.items():
            color, marker, label = STYLE[cond]
            days = sorted({r["day"] for r in recs if r["kind"] == kind})
            ys = [recall([r for r in recs if r["day"] == d and r["kind"] == kind]) for d in days]
            if not days:
                continue
            ax.plot(days, ys, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1.5, label=label)
            ax.annotate(label.split()[0], (days[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_title({"direct": "Direct questions", "indirect": "Indirect questions",
                      "chain": "Chain questions (2 memories)"}[kind], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("day", color=MUTED)
        ax.set_xticks(sorted({r["day"] for recs in runs.values() for r in recs}))
        ax.set_xlim(0.5, 11)
        ax.set_facecolor("#fcfcfb")
        ax.set_ylim(-0.03, 1.05)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("recall", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Experiment 001: what the model still knows the next morning", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    out = HERE / "results" / "recall.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    return out


if __name__ == "__main__":
    runs = load()
    table(runs)
    print(f"\nsaved {plot(runs)}")
