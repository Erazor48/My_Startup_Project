"""Cost of the full-context reader along a conversation: prompt tokens, measured prefill and peak RAM, next to RAG and cards.

    uv run python cost_curve.py tokens                       # tokenizer only, every session count, no model
    uv run python cost_curve.py watch                        # all points, one child process each, with the stop rules
    uv run python cost_curve.py point --conv 0 --sessions 5  # one point (what `watch` runs)

The full-context prompt is the whole history, never truncated (locomo_run.LocoFull cuts at 16 000 tokens, which hides the cost).
Prefill alone is timed (the logits of the last token only, as a real inference does; the full-vocabulary logits of a long prompt
would add gigabytes that no user pays). Stop rules, set in advance: a single prefill above 10 min or a peak RSS above 22 GB ends
the point and the next ones of that conversation are not run ("not measurable on this laptop").
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import locomo

HERE = Path(__file__).parent
OUT = HERE / "runs" / "cost_curve"
POINTS = {0: [1, 5, 10, 19], 2: [1, 10, 20, 32]}  # LoCoMo index 0 = conv-26, 2 = conv-41
CARDS = {0: "oracle_haiku", 2: "local_qwen35_4b_conv41"}
N_QUESTIONS = 5
MAX_PREFILL_S = 600
MAX_RSS_GB = 22


def history(conv: locomo.Conversation, n: int) -> str:
    return "\n".join(f"({locomo.date_only(s.date)}) {speaker}: {text}" for s in conv.sessions[:n] for speaker, text in s.turns)


def questions_at(conv: locomo.Conversation, n: int) -> list:
    pool = [q for q in conv.questions if q.answerable_after(n)]
    keep = sorted(random.Random(0).sample(range(len(pool)), min(N_QUESTIONS, len(pool))))
    return [pool[i] for i in keep]


def tokens() -> None:
    from transformers import AutoTokenizer

    from lm import MODEL
    from locomo_run import ANSWER_SYSTEM

    tok = AutoTokenizer.from_pretrained(MODEL)
    result = {}
    for index, points in POINTS.items():
        conv = locomo.load(index)
        for n in points:
            counts = []
            for q in questions_at(conv, n):
                prompt = f"Past conversations (with their dates):\n{history(conv, n)}\n\n{q.question}"
                text = tok.apply_chat_template([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                counts.append(len(tok(text).input_ids))
            result[f"{conv.sample_id}:{n}"] = {"median_tokens": statistics.median(counts), "tokens": counts}
            print(conv.sample_id, n, "sessions", counts, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokens.json").write_text(json.dumps(result, indent=1))


def prefill(lm, prompt: str, log) -> tuple[int, float]:
    import torch

    from lm import reset_peak_rss

    ids = lm.tokenizer(lm.render([{"role": "user", "content": prompt}]), return_tensors="pt")
    reset_peak_rss()
    log.write(f"start {time.time()}\n")
    log.flush()
    t = time.time()
    with torch.no_grad():
        lm.model(**ids, use_cache=True, logits_to_keep=1)
    s = time.time() - t
    log.write(f"done {s}\n")
    log.flush()
    return ids.input_ids.shape[1], s


def point(index: int, n: int) -> None:
    from lm import LM, peak_rss_mb
    from locomo_run import ANSWER_SYSTEM, LocoRag
    from screen import NOTES_PROMPT, night_cards
    from second_conv import CardRag

    conv = locomo.load(index)
    lm = LM(system=ANSWER_SYSTEM)
    qs = questions_at(conv, n)
    raw = LocoRag(lm, 10)
    cards = CardRag(lm, {i: night_cards(CARDS[index], i, None) for i in range(1, n + 1)})
    cards.top_k = 2
    for s in conv.sessions[:n]:
        raw.live_day(s)
        cards.live_day(s)
    prompts = {
        "raw_k10": [f"Things said in past conversations (with their dates):\n" + "\n".join(raw.retrieve(q.question)) + f"\n\n{q.question}" for q in qs],
        "cards_top2": [NOTES_PROMPT.format(notes="\n".join(cards.retrieve(q.question)), question=q.question) for q in qs],
        "full_context": [f"Past conversations (with their dates):\n{history(conv, n)}\n\n{q.question}" for q in qs],
    }
    result = {"conv": conv.sample_id, "sessions": n, "questions": [q.question for q in qs]}
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"progress_{conv.sample_id}_{n}.log", "w")
    for label, ps in prompts.items():
        rows = [prefill(lm, p, log) for p in ps]
        result[label] = {"prompt_tokens": [r[0] for r in rows], "prefill_s": [round(r[1], 3) for r in rows],
                         "median_tokens": statistics.median(r[0] for r in rows), "median_prefill_s": round(statistics.median(r[1] for r in rows), 3),
                         "peak_rss_mb": round(peak_rss_mb())}
        print(conv.sample_id, n, label, result[label], flush=True)
        (OUT / f"{conv.sample_id}_{n}.json").write_text(json.dumps(result, indent=1))


def rss_gb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def watch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verdict = {}
    for index, points in POINTS.items():
        sample = locomo.load(index).sample_id
        blocked = None
        for n in points:
            key = f"{sample}:{n}"
            if blocked:
                verdict[key] = f"not measurable on this laptop (stopped at {blocked})"
                continue
            progress = OUT / f"progress_{sample}_{n}.log"
            progress.unlink(missing_ok=True)
            child = subprocess.Popen([sys.executable, __file__, "point", "--conv", str(index), "--sessions", str(n)])
            peak, reason = 0.0, None
            while child.poll() is None:
                time.sleep(2)
                peak = max(peak, rss_gb(child.pid))
                lines = progress.read_text().splitlines() if progress.exists() else []
                if peak > MAX_RSS_GB:
                    reason = f"RSS {peak:.1f} GB > {MAX_RSS_GB}"
                elif lines and lines[-1].startswith("start") and time.time() - float(lines[-1].split()[1]) > MAX_PREFILL_S:
                    reason = f"one prefill > {MAX_PREFILL_S} s"
                if reason:
                    child.kill()
                    child.wait()
                    break
            verdict[key] = {"stopped": reason, "watcher_peak_rss_gb": round(peak, 1)} if reason else {"ok": child.returncode == 0, "watcher_peak_rss_gb": round(peak, 1)}
            print(key, verdict[key], flush=True)
            if reason or child.returncode != 0:
                blocked = n
            (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tokens", "watch", "point"])
    p.add_argument("--conv", type=int)
    p.add_argument("--sessions", type=int)
    a = p.parse_args()
    {"tokens": tokens, "watch": watch, "point": lambda: point(a.conv, a.sessions)}[a.mode]()
