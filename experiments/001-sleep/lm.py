from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"
SYSTEM = "You are the user's personal assistant. Answer in one short sentence."


@dataclass
class Reply:
    text: str
    prompt_tokens: int


class LM:
    def __init__(self, model_name: str = MODEL):
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

    def render(self, messages: list[dict], system: str = SYSTEM, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, *messages],
            add_generation_prompt=add_generation_prompt, tokenize=False, enable_thinking=False,
        )

    def chat(self, messages: list[dict], max_new_tokens: int = 32) -> Reply:
        ids = self.tokenizer(self.render(messages), return_tensors="pt")
        n_prompt = ids.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(**ids, **self._stop(max_new_tokens))
        text = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
        return Reply(text, n_prompt)

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
