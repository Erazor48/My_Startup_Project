"""C1 (the whole history in the prompt, never truncated) on the questions of conv-26 answerable in its 19 sessions.

    uv run python c1_full.py --conv 0 --sessions 19

Resumable: every answer is appended to runs/screen/c1_full_<tag>.jsonl and a rerun skips what is done. Same questions, filter and
reader settings as screen.py's c0s19_ files. One prefill per question (generation alone, the prefill timed to the first token),
the logits of the last token only. Stop rules, set in advance: a peak RSS above 22 GB, or a projected total above 6 h once the
first 10 questions are done. On a stop, the partial file keeps its n; nothing is extrapolated.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

import locomo
from lm import LM, Reply, peak_rss_mb, reset_peak_rss
from locomo_run import ANSWER_SYSTEM

HERE = Path(__file__).parent
OUT = HERE / "runs" / "screen"
MAX_RSS_MB = 22 * 1024
MAX_PROJECTED_S = 6 * 3600
PROJECT_AFTER = 10


class FirstToken(LogitsProcessor):
    """Called once the prefill is over: the time from the start to this call is the prefill."""

    def __init__(self, start: float):
        self.start, self.prefill_s = start, None

    def __call__(self, input_ids, scores):
        if self.prefill_s is None:
            self.prefill_s = time.time() - self.start
        return scores


def answer(lm: LM, history: str, question: str) -> Reply:
    prompt = f"Past conversations (with their dates):\n{history}\n\n{question}"
    ids = lm.tokenizer(lm.render([{"role": "user", "content": prompt}]), return_tensors="pt")
    n_prompt = ids.input_ids.shape[1]
    reset_peak_rss()
    t = time.time()
    first = FirstToken(t)
    with torch.no_grad():
        out = lm.model.generate(**ids, **lm._stop(24), logits_processor=LogitsProcessorList([first]))
    text = lm.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
    return Reply(text, n_prompt, round(first.prefill_s, 3), round(peak_rss_mb()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=19)
    args = parser.parse_args()

    conv = locomo.load(args.conv)
    history = "\n".join(f"({locomo.date_only(s.date)}) {speaker}: {text}" for s in conv.sessions[:args.sessions] for speaker, text in s.turns)
    questions = [q for q in conv.questions if q.answerable_after(args.sessions)]
    path = OUT / f"c1_full_c{args.conv}s{args.sessions}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    print(f"{len(questions)} questions, {len(done)} already done", flush=True)
    lm = LM(system=ANSWER_SYSTEM)
    started = time.time()
    for i, q in enumerate(questions[len(done):], len(done)):
        t = time.time()
        reply = answer(lm, history, q.question)
        record = {"category": q.category, "question": q.question, "expected": q.answer, "answer": reply.text, "f1": locomo.question_f1(q, reply.text),
                  "prompt_tokens": reply.prompt_tokens, "answer_s": round(time.time() - t, 2), "prefill_s": reply.prefill_s, "peak_rss_mb": reply.peak_rss_mb}
        done.append(record)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        mean_s = statistics.mean(r["answer_s"] for r in done)
        print(f"{len(done)}/{len(questions)} F1 {record['f1']:.2f} (running {100 * statistics.mean(r['f1'] for r in done):.1f}) "
              f"prefill {reply.prefill_s}s answer {record['answer_s']}s rss {reply.peak_rss_mb} MB, projected {mean_s * len(questions) / 3600:.1f} h", flush=True)
        if reply.peak_rss_mb > MAX_RSS_MB:
            print(f"STOP: peak RSS {reply.peak_rss_mb} MB > {MAX_RSS_MB}", flush=True)
            return
        if len(done) >= PROJECT_AFTER and mean_s * len(questions) > MAX_PROJECTED_S:
            print(f"STOP: projected {mean_s * len(questions) / 3600:.1f} h > 6 h", flush=True)
            return
    print(f"finished, n={len(done)}, {(time.time() - started) / 60:.0f} min this session", flush=True)


if __name__ == "__main__":
    main()
