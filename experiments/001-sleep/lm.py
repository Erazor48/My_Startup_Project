import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"
SYSTEM = "You are the user's personal assistant. Answer in one short sentence."


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    prefill_s: float = 0.0
    peak_rss_mb: float = 0.0


def reset_peak_rss() -> None:
    """Restart the kernel's peak-RSS counter (VmHWM) of this process."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def peak_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


class LM:
    def __init__(self, model_name: str = MODEL, system: str = SYSTEM):
        self.system = system
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()
        # The generation config does not stop at the end of a chat turn: without this the model keeps
        # writing the next turns of the conversation.
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _stop(self, max_new_tokens: int) -> dict:
        return {
            "max_new_tokens": max_new_tokens, "do_sample": False,
            "eos_token_id": self.end_of_turn, "pad_token_id": self.tokenizer.pad_token_id,
        }

    def render(self, messages: list[dict], system: str | None = None, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system if system is None else system}, *messages],
            add_generation_prompt=add_generation_prompt, tokenize=False, enable_thinking=False,
        )

    def chat(self, messages: list[dict], max_new_tokens: int = 32) -> Reply:
        ids = self.tokenizer(self.render(messages), return_tensors="pt")
        n_prompt = ids.input_ids.shape[1]
        reset_peak_rss()
        with torch.no_grad():
            t = time.time()
            self.model(**ids, use_cache=True)  # the prefill alone, timed apart from the decoding
            prefill_s = time.time() - t
            out = self.model.generate(**ids, **self._stop(max_new_tokens))
        text = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
        return Reply(text, n_prompt, round(prefill_s, 3), round(peak_rss_mb()))

    def choose(self, prompt: str, options: list[str], system: str) -> int:
        """Index of the option whose first token is the most likely start of the answer (no generation)."""
        ids = self.tokenizer(self.render([{"role": "user", "content": prompt}], system=system), return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        first_tokens = [self.tokenizer.encode(o, add_special_tokens=False)[0] for o in options]
        return int(logits[first_tokens].argmax())

    def complete_batch(self, prompts: list[str], system: str, max_new_tokens: int = 128) -> list[str]:
        """Greedy single-turn completions for several prompts at once (left padding)."""
        self.tokenizer.padding_side = "left"
        texts = [self.render([{"role": "user", "content": p}], system=system) for p in prompts]
        ids = self.tokenizer(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model.generate(**ids, **self._stop(max_new_tokens))
        n = ids.input_ids.shape[1]
        return [self.tokenizer.decode(o[n:], skip_special_tokens=True).strip() for o in out]
