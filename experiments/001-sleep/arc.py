"""General skills check: ARC-Easy accuracy of the base model, or of the model after its nights of sleep.

The answer is the option letter with the highest next-token logit, so formatting habits picked up during
sleep (short sentences) do not count as lost knowledge.

    uv run python arc.py                                  # base model
    uv run python arc.py adapters/c4_sleep_replay_seed0   # after 10 nights
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel

from lm import LM

OUT = Path(__file__).parent / "runs" / "arc"
SYSTEM = "Answer the multiple-choice question with the letter of the correct option."


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
        labels = q["choices"]["label"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, q["choices"]["text"]))
        correct.append(labels[lm.choose(f"{q['question']}\n{options}", labels, SYSTEM)] == q["answerKey"])

    name = args.adapter.name if args.adapter else "base"
    accuracy = sum(correct) / len(correct)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"model": name, "accuracy": accuracy, "correct": correct}))
    print(f"{name}: ARC-Easy {accuracy:.1%} ({sum(correct)}/{len(correct)})")


if __name__ == "__main__":
    main()
