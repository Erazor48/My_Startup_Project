"""Regenerate the README figures and tables of Hypnos from the committed run files. No model is loaded.

    uv run --with matplotlib python figures.py                       # paths found automatically
    uv run --with matplotlib python figures.py --exp PATH --out PATH  # explicit paths

Reads experiments/001-sleep/runs/{locomo,screen,arc,arc_cloze}/*.json and writes two PNGs to --out (default: img/
next to this file). Prints the README tables (F1 x 100 with a 95 % bootstrap interval over questions, median prompt
tokens, median prefill time) so every number in the README can be checked against the files.

The decisive control (RAG top-2 on the same cards, no adapter) is drawn as "pending" until a file matching
runs/screen/*ragref2_seed0.json exists; it is picked up automatically once it does.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE.parents[1] / "code" / "experiments" / "001-sleep",  # hypnos/docs/publication/figures.py (draft layout)
    HERE.parent / "experiments" / "001-sleep",  # <repo>/docs/figures.py (repository layout)
]

# Okabe-Ito palette (colour-blind safe): where the memory lives at answer time.
KIND_COLOR = {"none": "#999999", "weights": "#0072B2", "prompt": "#E69F00", "hybrid": "#009E73"}
KIND_LABEL = {
    "none": "no memory",
    "weights": "memory in the weights (LoRA), 74-token prompt",
    "prompt": "memory in the prompt (RAG)",
    "hybrid": "weights + 2 retrieved cards",
}

# (file under runs/, label, kind). Order = order in the tables and the bar chart.
CONV26 = [
    ("locomo/c0_none_conv0.json", "No memory", "none"),
    ("locomo/c3_lora_raw_conv0.json", "Naive LoRA on raw text", "weights"),
    ("locomo/c4_sleep_replay_conv0.json", "Sleep V1 (2B cards + replay)", "weights"),
    ("locomo/c5_sleep_integrate_conv0.json", "Sleep V2 (V1 + integration)", "weights"),
    ("locomo/diag_d_onepass_2b_conv0.json", "2B cards, one training pass", "weights"),
    ("locomo/diag_d_onepass_oracle_conv0.json", "Haiku cards, one training pass", "weights"),
    ("locomo/c2_rag_k4_conv0.json", "RAG top-4, raw turns", "prompt"),
    ("locomo/c2_rag_k10_conv0.json", "RAG top-10, raw turns", "prompt"),
    ("locomo/diag_a_2b_rag_k10_conv0.json", "RAG top-10, 2B cards", "prompt"),
    ("locomo/diag_a_oracle_rag_k10_conv0.json", "RAG top-10, Haiku cards", "prompt"),
]
S8 = [
    ("screen/baseline_seed0.json", "2B cards in the weights", "weights"),
    ("screen/oracle_seed0.json", "Haiku cards in the weights", "weights"),
    ("screen/oracle_hybrid2_seed0.json", "Hybrid: weights + 2 Haiku cards", "hybrid"),
    ("screen/*ragref2_seed0.json", "Control: RAG top-2, Haiku cards, no LoRA", "prompt"),
    ("screen/oracle_ragref10_seed0.json", "RAG top-10, Haiku cards, no LoRA", "prompt"),
]
ARC = [("base.json", "Base model"), ("locomo_c3_lora_raw_conv0.json", "Naive LoRA on raw text"),
       ("locomo_c4_sleep_replay_conv0.json", "Sleep V1"), ("locomo_c5_sleep_integrate_conv0.json", "Sleep V2")]


def find_exp(arg: Path | None) -> Path:
    for path in ([arg] if arg else CANDIDATES):
        if path and (path / "runs" / "locomo").is_dir():
            return path
    raise SystemExit(f"experiments/001-sleep not found; pass --exp (tried: {', '.join(map(str, CANDIDATES))})")


def bootstrap(values: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval of the mean, resampling questions."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(runs: Path, spec: list[tuple[str, str, str]]) -> list[dict]:
    rows = []
    for pattern, label, kind in spec:
        matches = sorted(runs.glob(pattern))
        if not matches:
            rows.append({"label": label, "kind": kind, "file": pattern, "pending": True})
            continue
        records = json.loads(matches[0].read_text())["records"]
        f1 = [r["f1"] for r in records]
        lo, hi = bootstrap(f1)
        rows.append({
            "label": label, "kind": kind, "file": str(matches[0].relative_to(runs)), "pending": False, "n": len(records),
            "f1": 100 * mean(f1), "lo": 100 * lo, "hi": 100 * hi,
            "tokens": median(r["prompt_tokens"] for r in records),
            "prefill": median(r["prefill_s"] for r in records),
            "questions": [r["question"] for r in records], "f1s": f1,
        })
    return rows


def table(title: str, rows: list[dict]) -> None:
    print(f"\n### {title}\n")
    print("| Condition | Memory at answer time | F1 | 95 % CI | Median prompt tokens | Median prefill (s) | File |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["pending"]:
            print(f"| {r['label']} | {KIND_LABEL[r['kind']]} | [TO FILL after control] | | | | `{r['file']}` |")
        else:
            print(f"| {r['label']} | {KIND_LABEL[r['kind']]} | {r['f1']:.1f} | {r['lo']:.1f}-{r['hi']:.1f} "
                  f"| {r['tokens']:.0f} | {r['prefill']:.2f} | `{r['file']}` (n={r['n']}) |")


def paired(rows: list[dict], a: str, b: str) -> None:
    """Paired difference b - a on the same questions, with a bootstrap interval."""
    ra, rb = (next(r for r in rows if r["label"] == x and not r["pending"]) for x in (a, b))
    theirs = dict(zip(rb["questions"], rb["f1s"]))
    diffs = [theirs[q] - f for q, f in zip(ra["questions"], ra["f1s"]) if q in theirs]
    lo, hi = bootstrap(diffs)
    print(f"- paired, {len(diffs)} questions: {b} minus {a} = {100 * mean(diffs):+.1f} F1 (95 % CI {100 * lo:+.1f} to {100 * hi:+.1f})")


def night_cost(exp: Path, conv: list[dict]) -> None:
    by = {r["label"]: r for r in conv}
    for name, label in (("c4_sleep_replay", "Sleep V1"), ("c5_sleep_integrate", "Sleep V2")):
        nights = json.loads((exp / "runs" / "locomo" / f"{name}_conv0.json").read_text())["nights"]
        s = [n["sleep_s"] for n in nights]
        print(f"- {label}: {len(s)} nights, {mean(s):.0f} s per night on average ({mean(s) / 60:.1f} min CPU), "
              f"{sum(s) / 60:.0f} min in total")
    night = mean(n["sleep_s"] for n in json.loads((exp / "runs/locomo/c4_sleep_replay_conv0.json").read_text())["nights"])
    weights, rag = by["Sleep V1 (2B cards + replay)"], by["RAG top-10, raw turns"]
    saved = rag["prefill"] - weights["prefill"]
    print(f"- prefill saved per question vs RAG top-10 raw: {rag['prefill']:.2f} - {weights['prefill']:.2f} = {saved:.2f} s; "
          f"one night pays back after about {night / saved:.0f} questions per session")


def arc(exp: Path) -> None:
    print("\n### ARC-Easy, 200 questions, after the 19 LoCoMo nights\n")
    print("| Model | letter | cloze |")
    print("|---|---|---|")
    for file, label in ARC:
        acc = [json.loads((exp / "runs" / d / file).read_text())["accuracy"] for d in ("arc", "arc_cloze")]
        print(f"| {label} | {100 * acc[0]:.1f} | {100 * acc[1]:.1f} |")


def bars(conv: list[dict], s8: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [len(conv), len(s8) + 1.5]})
    panels = ((axes[0], conv, "LoCoMo conv-26, all 19 sessions, 152 questions"),
              (axes[1], s8, "Screening S8: sessions 1-8, 66 questions"))
    for ax, rows, title in panels:
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            if r["pending"]:
                ax.barh(yi, 0.0, color="none")
                ax.text(1, yi, "pending: decisive control, not run yet", va="center", fontsize=9, style="italic", color="#555")
                continue
            ax.barh(yi, r["f1"], color=KIND_COLOR[r["kind"]], height=0.7)
            ax.errorbar(r["f1"], yi, xerr=[[r["f1"] - r["lo"]], [r["hi"] - r["f1"]]], fmt="none", ecolor="#333", capsize=3, lw=1)
            ax.text(r["hi"] + 1, yi, f"{r['f1']:.1f}  ({r['tokens']:.0f} tok)", va="center", fontsize=9)
        ax.set_yticks(y, [r["label"] for r in rows], fontsize=9)
        ax.set_xlim(0, 80)
        ax.set_xlabel("F1 (x100), bars = 95 % bootstrap CI over questions")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in KIND_COLOR]
    fig.legend(handles, [KIND_LABEL[k] for k in KIND_COLOR], loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("F1 by condition (Qwen3.5-2B on CPU, one conversation, one seed); median prompt tokens in brackets", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def spread_labels(points: list[tuple[float, float]], gap: float = 3.2, same_x: float = 20) -> list[float]:
    """Label heights: points sharing (almost) the same x get labels at least `gap` F1 points apart."""
    ys = [y for _, y in points]
    order = sorted(range(len(points)), key=lambda i: (round(points[i][0] / same_x), points[i][1]))
    for prev, cur in zip(order, order[1:]):
        if abs(points[cur][0] - points[prev][0]) < same_x and ys[cur] - ys[prev] < gap:
            ys[cur] = ys[prev] + gap
    return ys


def scatter(conv: list[dict], s8: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    shown = [(r, "o") for r in conv if not r["pending"]] + [(r, "s") for r in s8 if not r["pending"]]
    label_y = spread_labels([(r["tokens"], r["f1"]) for r, _ in shown])
    for (r, marker), ly in zip(shown, label_y):
        ax.errorbar(r["tokens"], r["f1"], yerr=[[r["f1"] - r["lo"]], [r["hi"] - r["f1"]]], fmt=marker, ms=7,
                    color=KIND_COLOR[r["kind"]], ecolor=KIND_COLOR[r["kind"]], elinewidth=0.8, capsize=2, alpha=0.85)
        crowded = r["tokens"] < 100
        ax.annotate(r["label"], (r["tokens"], r["f1"]), xytext=(r["tokens"] + (40 if crowded else 8), ly + 0.6),
                    textcoords="data", fontsize=8, va="center",
                    arrowprops={"arrowstyle": "-", "color": "#999", "lw": 0.6} if crowded else None)
    for marker, name in (("o", "conv-26, 152 questions"), ("s", "S8 screening, 66 questions")):
        ax.plot([], [], marker, color="#444", ls="none", label=name)
    for kind in KIND_COLOR:
        ax.plot([], [], "o", color=KIND_COLOR[kind], ls="none", label=KIND_LABEL[kind])
    if any(r["pending"] for r in s8):
        ax.text(0.01, 0.98, "Not shown: RAG top-2 on the same cards without LoRA (control pending)", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, style="italic", color="#555")
    ax.set_xlabel("Median prompt tokens per question (memory + question + system prompt)")
    ax.set_ylabel("F1 (x100), bars = 95 % bootstrap CI")
    ax.set_xlim(40, 560)
    ax.set_ylim(0, 75)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.04))
    ax.set_title("Accuracy against prompt cost (Qwen3.5-2B, CPU, one seed). Up and left is better.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, help="path to experiments/001-sleep")
    parser.add_argument("--out", type=Path, default=HERE / "img")
    args = parser.parse_args()
    exp = find_exp(args.exp)
    runs = exp / "runs"
    conv, s8 = load(runs, CONV26), load(runs, S8)

    print(f"Reading {runs}")
    table("LoCoMo conv-26 (152 questions)", conv)
    table("Screening S8 (sessions 1-8 of conv-26, 66 questions)", s8)
    print()
    paired(s8, "Haiku cards in the weights", "Hybrid: weights + 2 Haiku cards")
    paired(s8, "Hybrid: weights + 2 Haiku cards", "RAG top-10, Haiku cards, no LoRA")
    if not next(r for r in s8 if r["label"].startswith("Control"))["pending"]:
        paired(s8, "Control: RAG top-2, Haiku cards, no LoRA", "Hybrid: weights + 2 Haiku cards")
    print("\n### Night cost (conv-26)\n")
    night_cost(exp, conv)
    arc(exp)

    args.out.mkdir(parents=True, exist_ok=True)
    bars(conv, s8, args.out / "f1_by_condition.png")
    scatter(conv, s8, args.out / "f1_vs_tokens.png")
    print(f"\nWrote {args.out / 'f1_by_condition.png'} and {args.out / 'f1_vs_tokens.png'}")


if __name__ == "__main__":
    main()
