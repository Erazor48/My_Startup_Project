"""Summarize runs/locomo/*.json: F1 per category, cost per question, the cost-vs-history curve and the price of a night.

    uv run python analyze_locomo.py
"""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from locomo import CATEGORIES

HERE = Path(__file__).parent
ORDER = ["c0_none", "c1_full_context", "c2_rag_k4", "c2_rag_k10", "c3_lora_raw", "c4_sleep_replay", "c5_sleep_integrate"]
LABEL = {
    "c0_none": "C0 no memory", "c1_full_context": "C1 full context", "c2_rag_k4": "C2 RAG k=4", "c2_rag_k10": "C2 RAG k=10",
    "c3_lora_raw": "LoRA on raw text", "c4_sleep_replay": "C4 sleep+replay", "c5_sleep_integrate": "C5 sleep+integration",
}
COLOR = {
    "c0_none": "#2a78d6", "c1_full_context": "#eb6834", "c2_rag_k4": "#1baf7a", "c2_rag_k10": "#0e7a54",
    "c3_lora_raw": "#eda100", "c4_sleep_replay": "#e87ba4", "c5_sleep_integrate": "#008300",
}


def load() -> dict[tuple[str, int], dict]:
    runs = {}
    for path in sorted((HERE / "runs" / "locomo").glob("*_conv*.json")):
        condition, conv = path.stem.rsplit("_conv", 1)
        runs[(condition, int(conv))] = json.loads(path.read_text())
    return runs


def f1(records: list[dict]) -> float:
    return 100 * mean(r["f1"] for r in records) if records else float("nan")


def arc(name: str, folder: str) -> str:
    path = HERE / "runs" / folder / f"locomo_{name}.json"
    return f"{100 * json.loads(path.read_text())['accuracy']:.1f}" if path.exists() else "-"


def main() -> None:
    runs = load()
    convs = sorted({c for _, c in runs})
    for conv in convs:
        print(f"\n## Conversation {conv} - final F1 (x100) by category, cost per question\n")
        cats = list(CATEGORIES.items())
        head = ["condition", "n", *(n for _, n in cats), "all", "prompt tok", "prefill s", "answer s", "peak RSS MB"]
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for cond in ORDER:
            run = runs.get((cond, conv))
            if not run:
                continue
            rs = run["records"]
            by_cat = [f"{f1([r for r in rs if r['category'] == c]):.1f}" for c, _ in cats]
            note = " (truncated)" if run.get("truncated") else ""
            print(f"| {LABEL[cond]}{note} | {len(rs)} | " + " | ".join(by_cat) + f" | **{f1(rs):.1f}** | {mean(r['prompt_tokens'] for r in rs):.0f}"
                  f" | {mean(r['prefill_s'] for r in rs):.2f} | {mean(r['answer_s'] for r in rs):.1f} | {max(r['peak_rss_mb'] for r in rs)} |")

        print(f"\n## Conversation {conv} - cost and F1 as the history grows (questions whose evidence is in the first n sessions)\n")
        print("| condition | sessions | n | F1 | prompt tok | prefill s | peak RSS MB |\n|---|---|---|---|---|---|---|")
        for cond in ORDER:
            run = runs.get((cond, conv))
            for n, rs in (run or {}).get("curve", {}).items():
                if rs:
                    print(f"| {LABEL[cond]} | {n} | {len(rs)} | {f1(rs):.1f} | {mean(r['prompt_tokens'] for r in rs):.0f}"
                          f" | {mean(r['prefill_s'] for r in rs):.2f} | {max(r['peak_rss_mb'] for r in rs)} |")

        print(f"\n## Conversation {conv} - the price of the nights\n")
        print("| condition | nights | total CPU min | mean min/night | last night min | peak RSS MB | cards | ARC letters | ARC cloze |\n|---|---|---|---|---|---|---|---|---|")
        for cond in ("c3_lora_raw", "c4_sleep_replay", "c5_sleep_integrate"):
            run = runs.get((cond, conv))
            if not run:
                continue
            nights = run["nights"]
            secs = [n["sleep_s"] for n in nights]
            name = f"{cond}_conv{conv}"
            print(f"| {LABEL[cond]} | {len(nights)} | {sum(secs) / 60:.0f} | {mean(secs) / 60:.1f} | {secs[-1] / 60:.1f} | {max(n['peak_rss_mb'] for n in nights)}"
                  f" | {sum(n.get('cards', 0) for n in nights)} | {arc(name, 'arc')} | {arc(name, 'arc_cloze')} |")

        rag, sleepers = runs.get(("c2_rag_k4", conv)), ["c4_sleep_replay", "c5_sleep_integrate"]
        if rag:
            print(f"\n## Conversation {conv} - verdict against the card's criteria\n")
            rag_tokens = mean(r["prompt_tokens"] for r in rag["records"])
            for cond in sleepers:
                run = runs.get((cond, conv))
                if not run:
                    continue
                share = f1(run["records"]) / f1(rag["records"])
                tokens = mean(r["prompt_tokens"] for r in run["records"])
                multi = lambda r: [x for x in r["records"] if x["category"] == 1]
                print(f"- {LABEL[cond]}: {100 * share:.0f}% of the RAG F1, {rag_tokens / tokens:.1f}x fewer prompt tokens than RAG k=4; "
                      f"multi-hop F1 {f1(multi(run)):.1f} vs RAG {f1(multi(rag)):.1f}")

    plot(runs, convs)


def plot(runs: dict, convs: list[int]) -> None:
    (HERE / "results").mkdir(exist_ok=True)
    for conv in convs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for cond in ORDER:
            curve = runs.get((cond, conv), {}).get("curve")
            if not curve:
                continue
            xs = sorted(int(n) for n in curve)
            for ax, key in zip(axes, ("prompt_tokens", "prefill_s", "f1")):
                ys = [mean(r[key] for r in curve[str(n)]) * (100 if key == "f1" else 1) for n in xs]
                ax.plot(xs, ys, marker="o", color=COLOR[cond], label=LABEL[cond])
        for ax, title in zip(axes, ("prompt tokens per question", "prefill latency per question (s)", "F1 (x100)")):
            ax.set_title(title)
            ax.set_xlabel("sessions lived")
            ax.grid(alpha=0.3)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / "results" / f"locomo_cost_conv{conv}.png", dpi=130)


if __name__ == "__main__":
    main()
