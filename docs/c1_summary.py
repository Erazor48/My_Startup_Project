"""F1 of the full-context reader (C1) on conv-26 and its paired gaps to the Haiku-card and raw RAG runs.

    uv run python docs/c1_summary.py
"""

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

SCREEN = Path(__file__).resolve().parent.parent / "experiments" / "001-sleep" / "runs" / "screen"
OTHERS = {"cards top-2 (Haiku, c0s19_oracle_ragref2)": "c0s19_oracle_ragref2_seed0.json", "raw RAG top-10 (c0s19_oracle_ragref10)": "c0s19_oracle_ragref10_seed0.json"}


def bootstrap(values: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values)
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    c1 = [json.loads(line) for line in (SCREEN / "c1_full_c0s19.jsonl").read_text().splitlines()]
    f1 = [r["f1"] for r in c1]
    lo, hi = bootstrap(f1)
    print(f"C1 full context, conv-26: n={len(c1)}, F1 {100 * mean(f1):.1f} (95 % CI {100 * lo:.1f}-{100 * hi:.1f}), median tokens {median(r['prompt_tokens'] for r in c1):.0f}, "
          f"median prefill {median(r['prefill_s'] for r in c1):.1f} s, max peak RSS {max(r['peak_rss_mb'] for r in c1) / 1024:.1f} GB, total {sum(r['answer_s'] for r in c1) / 3600:.1f} h")
    for cat in sorted({r["category"] for r in c1}):
        sub = [r["f1"] for r in c1 if r["category"] == cat]
        print(f"  category {cat}: n={len(sub)}, F1 {100 * mean(sub):.1f}")
    mine = {r["question"]: r for r in c1}
    for label, name in OTHERS.items():
        other = json.loads((SCREEN / name).read_text())["records"]
        pairs = [(r["f1"], mine[r["question"]]["f1"], r["category"]) for r in other if r["question"] in mine]
        diffs = [a - b for a, b, _ in pairs]
        dlo, dhi = bootstrap(diffs)
        print(f"{label} minus C1: n={len(pairs)}, other F1 {100 * mean(a for a, _, _ in pairs):.1f}, {100 * mean(diffs):+.1f} (95 % CI {100 * dlo:+.1f} to {100 * dhi:+.1f})")
        for cat in sorted({c for _, _, c in pairs}):
            sub = [a - b for a, b, c in pairs if c == cat]
            print(f"  category {cat}: n={len(sub)}, {100 * mean(sub):+.1f}")


if __name__ == "__main__":
    main()
