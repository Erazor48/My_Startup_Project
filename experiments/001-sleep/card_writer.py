"""Local card writer: a small open-weight model writes the flashcards of each LoCoMo session, on CPU.

    uv run python card_writer.py --conv 0 --sessions 8

The only difference with oracle_cards.py is the writer: same PROMPT, same deduplication, same files
(cards_cache/<variant>/night_XX.json and all.json). Only Qwen/Qwen3.5-4B (Apache-2.0) is allowed. Thinking is off and decoding is
greedy. A reply that holds no card is retried once with a stricter instruction; failures are counted, never filled in.
Per session timing (prefill, generation, tokens, CPU time, peak RSS) goes to cards_cache/<variant>/timing.json, raw replies to raw/.
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import locomo
from lm import peak_rss_mb, reset_peak_rss
from oracle_cards import PROMPT, distinct

HERE = Path(__file__).parent
ALLOWED_MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW_TOKENS = 4096
# per variant: (max_new_tokens, repetition_penalty). v2 caps a looping reply (session 8 of S8 ran to 4096 tokens in v1).
SETTINGS = {"local_qwen35_4b": (4096, 1.0), "local_qwen35_4b_v2": (2048, 1.05)}
DEFAULT_SETTINGS = (2048, 1.0)  # v1 decoding with only the circuit breaker, for every other variant name (v2 did not pass its gate)
RETRY_SUFFIX = "\n\nReply with the JSON list only: start with [ and end with ]. Keep every answer short."


def check_model(model: str) -> None:
    if model != ALLOWED_MODEL:
        raise ValueError(f"model {model!r} is not allowed, only {ALLOWED_MODEL!r}")


def parse(text: str) -> tuple[list[dict], str]:
    """Cards of a reply and how they were read: 'strict' (a valid JSON list), 'tolerant' (objects salvaged one by one) or 'none'."""
    start, end = text.find("["), text.rfind("]")
    try:
        if start < 0 or end < start:
            raise ValueError
        items, mode = json.loads(text[start:end + 1]), "strict"
    except ValueError:
        items, mode = [], "tolerant"
        decoder, i = json.JSONDecoder(), max(start, 0)
        while (i := text.find("{", i)) >= 0:
            try:
                item, j = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            items.append(item)
            i = j
    cards = [{"q": str(c["q"]).strip(), "a": str(c["a"]).strip()} for c in items if isinstance(c, dict) and c.get("q") and c.get("a")]
    return cards, mode if cards else "none"


class Writer:
    def __init__(self, model_name: str = ALLOWED_MODEL, max_new_tokens: int = MAX_NEW_TOKENS, repetition_penalty: float = 1.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        check_model(model_name)
        self.torch = torch
        self.max_new_tokens, self.repetition_penalty = max_new_tokens, repetition_penalty
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def write(self, prompt: str) -> dict:
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = self.tokenizer(text, return_tensors="pt")
        n_prompt = ids.input_ids.shape[1]
        reset_peak_rss()
        cpu, wall = time.process_time(), time.time()
        with self.torch.no_grad():
            self.model(**ids, use_cache=True)  # the prefill alone, timed apart
            prefill_s = time.time() - wall
            out = self.model.generate(
                **ids, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=self.repetition_penalty,
                eos_token_id=self.end_of_turn, pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out[0, n_prompt:]
        return {
            "text": self.tokenizer.decode(new, skip_special_tokens=True).strip(), "prompt_tokens": n_prompt, "new_tokens": len(new),
            "truncated": len(new) >= self.max_new_tokens, "prefill_s": round(prefill_s, 2), "generate_s": round(time.time() - wall - prefill_s, 2),
            "wall_s": round(time.time() - wall, 2), "cpu_s": round(time.process_time() - cpu, 1), "peak_rss_mb": round(peak_rss_mb()),
        }


class OllamaProcesses:
    """CPU time and peak resident memory of the Ollama server and its runners, sampled from /proc (the model lives in the runner, not in this process)."""

    def __init__(self, server_name: str = "ollama"):
        self.server_name, self.peak_kb, self._stop = server_name, 0, threading.Event()

    @staticmethod
    def _stat(pid: int) -> tuple[str, int, int, int] | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            rss = next(int(line.split()[1]) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("VmRSS:"))
            return Path(f"/proc/{pid}/comm").read_text().strip(), int(fields[1]), int(fields[11]) + int(fields[12]), rss
        except (OSError, ValueError, StopIteration):
            return None

    def snapshot(self) -> tuple[float, int]:
        """(CPU seconds, resident kB) summed over every process named ollama and over its children (the runners)."""
        stats = {pid: s for pid in map(int, filter(str.isdigit, (p.name for p in Path("/proc").iterdir()))) if (s := self._stat(pid))}
        family = {pid for pid, s in stats.items() if s[0] == self.server_name}
        family |= {pid for pid, s in stats.items() if s[1] in family}
        ticks = sum(stats[pid][2] for pid in family)
        return ticks / os.sysconf("SC_CLK_TCK"), sum(stats[pid][3] for pid in family)

    def start(self) -> None:
        self.peak_kb, self._stop = 0, threading.Event()
        self.cpu0 = self.snapshot()[0]

        def poll() -> None:
            while not self._stop.is_set():
                self.peak_kb = max(self.peak_kb, self.snapshot()[1])
                self._stop.wait(0.5)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, int]:
        self._stop.set()
        self.thread.join()
        cpu, kb = self.snapshot()
        return cpu - self.cpu0, max(self.peak_kb, kb)


class OllamaWriter:
    """The same model as Writer (Qwen3.5-4B, Q4_K_M) served by Ollama: greedy, no thinking, same circuit breaker. CPU and RAM are those of the Ollama processes."""

    def __init__(self, tag: str, max_new_tokens: int, repetition_penalty: float, num_ctx: int = 4096, cpu_only: bool = False, url: str = "http://localhost:11434"):
        self.tag, self.url, self.processes = tag, url, OllamaProcesses()
        self.options = {"temperature": 0, "num_predict": max_new_tokens, "repeat_penalty": repetition_penalty, "num_ctx": num_ctx, "seed": 0}
        if cpu_only:  # Ollama offloads to the iGPU by itself (Vulkan); num_gpu 0 keeps every layer on the CPU
            self.options["num_gpu"] = 0
        self.max_new_tokens = max_new_tokens

    def write(self, prompt: str) -> dict:
        body = {"model": self.tag, "messages": [{"role": "user", "content": prompt}], "think": False, "stream": False, "options": self.options, "keep_alive": "30m"}
        request = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        self.processes.start()
        wall = time.time()
        with urllib.request.urlopen(request, timeout=3600) as response:
            reply = json.load(response)
        wall = time.time() - wall
        cpu_s, peak_kb = self.processes.stop()
        ns = 1e9
        return {
            "text": reply["message"]["content"].strip(), "prompt_tokens": reply["prompt_eval_count"], "new_tokens": reply["eval_count"],
            "truncated": reply["eval_count"] >= self.max_new_tokens, "prefill_s": round(reply["prompt_eval_duration"] / ns, 2),
            "generate_s": round(reply["eval_duration"] / ns, 2), "load_s": round(reply["load_duration"] / ns, 2), "wall_s": round(wall, 2),
            "tokens_per_s": round(reply["eval_count"] / (reply["eval_duration"] / ns), 2), "cpu_s": round(cpu_s, 1), "peak_rss_mb": round(peak_kb / 1024),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", metavar="TAG", help="write with this Ollama tag (Qwen3.5-4B Q4_K_M) instead of transformers bf16")
    parser.add_argument("--cpu-only", action="store_true", help="with --ollama: no layer on the iGPU")
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--sessions", type=int, help="only the first N sessions")
    parser.add_argument("--variant", default="local_qwen35_4b")
    parser.add_argument("--model", default=ALLOWED_MODEL)
    args = parser.parse_args()
    check_model(args.model)

    conv = locomo.load(args.conv)
    cache = HERE / "cards_cache" / args.variant
    (cache / "raw").mkdir(parents=True, exist_ok=True)
    timing_path = cache / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {"model": args.model, "sessions": {}}
    writer = None
    merged = []
    for night, session in enumerate(conv.sessions[:args.sessions], 1):
        path = cache / f"night_{night:02d}.json"
        date = locomo.date_only(session.date)
        if not path.exists():
            settings = SETTINGS.get(args.variant, DEFAULT_SETTINGS)
            writer = writer or (OllamaWriter(args.ollama, *settings, cpu_only=args.cpu_only) if args.ollama else Writer(args.model, *settings))
            conversation = "\n".join(f"{speaker}: {line}" for speaker, line in session.turns)
            prompt = PROMPT.format(date=date, conversation=conversation)
            attempts = []
            for suffix in ("", RETRY_SUFFIX):
                reply = writer.write(prompt + suffix)
                cards, mode = parse(reply["text"])
                attempts.append({**{k: v for k, v in reply.items() if k != "text"}, "parse": mode, "cards": len(cards)})
                (cache / "raw" / f"night_{night:02d}_try{len(attempts)}.txt").write_text(reply["text"])
                print(f"night {night} try {len(attempts)}: {mode}, {len(cards)} cards, {reply['new_tokens']} tokens, prefill {reply['prefill_s']} s, "
                      f"generation {reply['generate_s']} s, cpu {reply['cpu_s']} s, peak {reply['peak_rss_mb']} MB", flush=True)
                if cards:
                    break
            timing["sessions"][str(night)] = {"attempts": attempts, "failed": not cards}
            timing_path.write_text(json.dumps(timing, indent=1))
            if not cards:
                continue
            path.write_text(json.dumps(distinct(cards), indent=1))
        merged += [{"night": night, "date": date, **c} for c in json.loads(path.read_text())]
    failed = [n for n, s in timing["sessions"].items() if s["failed"]]
    (cache / "all.json").write_text(json.dumps(merged, indent=1))
    print(f"{len(merged)} cards -> {cache / 'all.json'}; failed sessions: {failed or 'none'}")


if __name__ == "__main__":
    main()
