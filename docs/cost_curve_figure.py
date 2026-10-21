"""Figure and table of the full-context cost curve (experiments/001-sleep/runs/cost_curve/*.json).

    uv run python docs/cost_curve_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "001-sleep" / "runs" / "cost_curve"
OUT = Path(__file__).resolve().parent / "img" / "cost_curve.png"
SERIES = [("full_context", "full context (whole history)", "tab:red"), ("raw_k10", "raw RAG top-10", "tab:blue"), ("cards_top2", "cards top-2", "tab:green")]
CONVS = [("conv-26", [1, 5, 10, 19]), ("conv-41", [1, 10, 20, 32])]


def main() -> None:
    verdict = json.loads((RUNS / "verdict.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    print("| Conversation | Sessions | Full-context tokens | Prefill (s) | Peak RAM (GB) | Raw RAG top-10 tokens / prefill (s) | Cards top-2 tokens / prefill (s) |")
    print("|---|---|---|---|---|---|---|")
    for col, (conv, points) in enumerate(CONVS):
        data = {}
        for n in points:
            path = RUNS / f"{conv}_{n}.json"
            if path.exists() and "full_context" in json.loads(path.read_text()):
                data[n] = json.loads(path.read_text())
        xs = sorted(data)
        for key, label, color in SERIES:
            for row, field in enumerate(("median_tokens", "median_prefill_s")):
                axes[row][col].plot(xs, [data[n][key][field] for n in xs], "o-", color=color, label=label)
        for n in points:
            if n in data:
                d = data[n]
                print(f"| {conv} | {n} | {d['full_context']['median_tokens']:.0f} | {d['full_context']['median_prefill_s']:.1f} | {d['full_context']['peak_rss_mb'] / 1024:.1f} "
                      f"| {d['raw_k10']['median_tokens']:.0f} / {d['raw_k10']['median_prefill_s']:.2f} | {d['cards_top2']['median_tokens']:.0f} / {d['cards_top2']['median_prefill_s']:.2f} |")
            else:
                print(f"| {conv} | {n} | not measurable on this laptop | | | | |  ({verdict.get(f'{conv}:{n}')})")
        axes[0][col].set_title(f"{conv} (median of 4-5 questions per point)")
        axes[1][col].set_xlabel("sessions lived")
    axes[0][0].set_ylabel("median prompt tokens")
    axes[1][0].set_ylabel("median prefill (s), Qwen3.5-2B, CPU")
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Cost of the reader's prompt along a conversation")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
