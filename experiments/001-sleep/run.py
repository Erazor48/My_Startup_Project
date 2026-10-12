"""Live through the simulated days with one memory condition and log every probe.

Each cycle: live day d -> sleep -> next morning, probe every fact said up to day d.

    uv run python run.py c0_none --seed 0
"""

import argparse
import json
import time
from pathlib import Path

from evaluate import score
from lm import LM
from memory import CONDITIONS as BASELINES
from sleep import CONDITIONS as SLEEPERS
from world import make_world

RUNS = Path(__file__).parent / "runs"
ADAPTERS = Path(__file__).parent / "adapters"
CONDITIONS = {**BASELINES, **SLEEPERS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--eval-days", type=int, nargs="*", help="days to probe (default: all)")
    args = parser.parse_args()

    world = make_world(n_days=args.days, seed=args.seed)
    memory = CONDITIONS[args.condition](LM())
    eval_days = set(args.eval_days or range(1, args.days + 1))
    first_day = {}
    for f in world.facts:
        first_day.setdefault(f.slot, f.day)
    kinds = ("direct", "indirect", "chain")

    records = []
    for day, turns in enumerate(world.days, 1):
        memory.live_day(turns)
        t = time.time()
        memory.sleep()
        sleep_s = time.time() - t
        if day not in eval_days:
            continue
        for probe in world.probes(day):
            t = time.time()
            reply = memory.answer(probe.question)
            records.append({
                "day": day,
                "kind": probe.kind,
                "slot": probe.slot,
                "fact_day": first_day[probe.slot],
                "updated": probe.stale is not None,
                "question": probe.question,
                "expected": probe.answer,
                "answer": reply.text,
                "result": score(reply.text, probe),
                "prompt_tokens": reply.prompt_tokens,
                "answer_s": round(time.time() - t, 2),
            })
        parts = []
        for kind in kinds:
            rs = [r for r in records if r["day"] == day and r["kind"] == kind]
            if rs:
                parts.append(f"{kind} {sum(r['result'] == 'correct' for r in rs)}/{len(rs)}")
        print(f"day {day:2d}  " + "  ".join(parts) + f"  (sleep {sleep_s:.0f}s)", flush=True)

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{args.condition}_seed{args.seed}.json"
    out.write_text(json.dumps({"args": vars(args), "sleep_log": getattr(memory, "log", []), "records": records}, indent=1))
    print(f"saved {out}")
    if args.condition in SLEEPERS:
        adapter = ADAPTERS / f"{args.condition}_seed{args.seed}"
        memory.lm.model.save_pretrained(adapter)
        print(f"saved {adapter}")


if __name__ == "__main__":
    main()
