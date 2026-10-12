"""ARC-Easy scored as cloze: no option letters, no chat template.

Each option is scored by the log-likelihood of its text after "Question: ... Answer:". Unlike arc.py, this
cannot be lowered by an adapter that merely unlearned the multiple-choice letter format.

    uv run python arc_cloze.py                                  # base model
    uv run python arc_cloze.py adapters/c4_sleep_replay_seed0   # after 10 nights
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel

from lm import LM

OUT = Path(__file__).parent / "runs" / "arc_cloze"


def option_logprob(lm: LM, question: str, option: str) -> float:
    prefix = lm.tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").input_ids
    answer = lm.tokenizer(f" {option}", add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prefix, answer], dim=1)
    with torch.no_grad():
        logprobs = lm.model(ids).logits[0, :-1].float().log_softmax(-1)
    n = prefix.shape[1]
    return logprobs[n - 1 :].gather(1, ids[0, n:, None]).sum().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", nargs="?", type=Path)
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args()

    lm = LM()
    if args.adapter:
        lm.model = PeftModel.from_pretrained(lm.model, args.adapter)
        lm.model.eval()
    questions = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=0).select(range(args.n))

    correct = []
    for q in questions:
        scores = [option_logprob(lm, q["question"], text) for text in q["choices"]["text"]]
        best = max(range(len(scores)), key=scores.__getitem__)
        correct.append(q["choices"]["label"][best] == q["answerKey"])

    name = args.adapter.name if args.adapter else "base"
    accuracy = sum(correct) / len(correct)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"model": name, "accuracy": accuracy, "correct": correct}))
    print(f"{name}: ARC-Easy cloze {accuracy:.1%} ({sum(correct)}/{len(correct)})")


if __name__ == "__main__":
    main()
