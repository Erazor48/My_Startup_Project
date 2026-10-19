"""Local card writer: a small open-weight model writes the flashcards of each LoCoMo session, on CPU.

    uv run python card_writer.py --conv 0 --sessions 8

The only difference with oracle_cards.py is the writer: same PROMPT, same deduplication, same files
(cards_cache/<variant>/night_XX.json and all.json). Only Qwen/Qwen3.5-4B (Apache-2.0) is allowed. Thinking is off and decoding is
greedy. A reply that holds no card is retried once with a stricter instruction; failures are counted, never filled in.
Per session timing (prefill, generation, tokens, CPU time, peak RSS) goes to cards_cache/<variant>/timing.json, raw replies to raw/.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import locomo
from lm import peak_rss_mb, reset_peak_rss
from oracle_cards import PROMPT, distinct

HERE = Path(__file__).parent
ALLOWED_MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW_TOKENS = 4096
# per variant: (max_new_tokens, repetition_penalty). v2 caps a looping reply (session 8 of S8 ran to 4096 tokens in v1).
SETTINGS = {"local_qwen35_4b": (4096, 1.0), "local_qwen35_4b_v2": (2048, 1.05)}
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
        check_model(model_name)
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
        with torch.no_grad():
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


def main() -> None:
    parser = argparse.ArgumentParser()
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
            writer = writer or Writer(args.model, *SETTINGS.get(args.variant, SETTINGS["local_qwen35_4b"]))
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
