"""Oracle cards: a strong model writes the flashcards of each LoCoMo session, to test whether card quality is the bottleneck.

    uv run python oracle_cards.py --conv 0

Only one model is allowed (Ethan's rule), checked against GET /v1/models before any call. The key is read from
~/.config/hypnos/anthropic.env and never printed or logged. Every call appends its tokens and cost to runs/logs/oracle_usage.jsonl.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import locomo

HERE = Path(__file__).parent
ALLOWED_MODEL = "claude-haiku-5-5"
ENV_FILE = Path.home() / ".config" / "hypnos" / "anthropic.env"
API = "https://api.anthropic.com/v1"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # dollars per token, prompts up to 100k tokens
BUDGET = 5.0
USAGE = HERE / "runs" / "logs" / "oracle_usage.jsonl"
PROMPT = """Below is a conversation between two people, held on {date}. Write flashcards for a long-term memory: questions that someone who did not hear the conversation will ask months later, each with a short answer.

Rules:
- Every question must stand alone: use the people's names, never "he", "she", "they", "the speaker" or "the event" without saying which.
- Answers are short (a few words), and repeat the name when the answer is a person.
- Turn every relative date ("yesterday", "last weekend", "next month", "last year") into an absolute date computed from {date}. Write "7 May 2023" when the day is known, "May 2023" or "2022" when only that is known.
- One card per fact. No duplicates, no card about small talk, greetings or feelings of the moment.
- For each event, write a card asking when it happened.
- Also write cards for what the conversation lets one infer: interests, traits, plans, preferences, family, places, hobbies, things owned, and what each person would likely want or do.
- Cover every concrete detail: names, titles, places, numbers, objects, colours, activities.

Answer with a JSON list only, no other text: [{{"q": "...", "a": "..."}}, ...]

Conversation of {date}:
{conversation}"""


def read_key() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no ANTHROPIC_API_KEY in {ENV_FILE}")


def request(path: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}/{path}", data=None if body is None else json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read())


def check_model(key: str, model: str) -> None:
    if model != ALLOWED_MODEL:
        raise ValueError(f"model {model!r} is not allowed, only {ALLOWED_MODEL!r}")
    listed = {m["id"] for m in request("models?limit=1000", key)["data"]}
    if model not in listed:
        raise ValueError(f"{model!r} is not served by /v1/models")


def spent() -> float:
    return sum(json.loads(line)["cost"] for line in USAGE.read_text().splitlines()) if USAGE.exists() else 0.0


def call(key: str, model: str, prompt: str, label: str) -> str:
    check_model(key, model)
    if spent() >= BUDGET:
        raise RuntimeError(f"budget of {BUDGET} dollars reached")
    reply = request("messages", key, {"model": model, "max_tokens": 8000, "messages": [{"role": "user", "content": prompt}]})
    usage = reply["usage"]
    cost = usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE, "a") as f:
        f.write(json.dumps({"label": label, "model": model, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "cost": cost}) + "\n")
    print(f"{label}: {usage['input_tokens']} in, {usage['output_tokens']} out, ${cost:.5f} (total ${spent():.5f})", flush=True)
    return "".join(block["text"] for block in reply["content"] if block["type"] == "text")


def parse(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"no JSON list in the reply: {text[:200]!r}")
    return [{"q": c["q"].strip(), "a": str(c["a"]).strip()} for c in json.loads(match.group(0)) if c.get("q") and c.get("a")]


def distinct(cards: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cards:
        if (c["q"].lower(), c["a"].lower()) not in seen:
            seen.add((c["q"].lower(), c["a"].lower()))
            out.append(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv", type=int, default=0)
    parser.add_argument("--sessions", type=int, help="only the first N sessions")
    parser.add_argument("--variant", default="oracle_haiku")
    parser.add_argument("--model", default=ALLOWED_MODEL)
    args = parser.parse_args()

    key = read_key()
    check_model(key, args.model)
    conv = locomo.load(args.conv)
    cache = HERE / "cards_cache" / args.variant
    cache.mkdir(parents=True, exist_ok=True)
    merged = []
    for night, session in enumerate(conv.sessions[:args.sessions], 1):
        path = cache / f"night_{night:02d}.json"
        date = locomo.date_only(session.date)
        if not path.exists():
            text = "\n".join(f"{speaker}: {line}" for speaker, line in session.turns)
            cards = parse(call(key, args.model, PROMPT.format(date=date, conversation=text), f"{args.variant} night {night}"))
            path.write_text(json.dumps(distinct(cards), indent=1))
        merged += [{"night": night, "date": date, **c} for c in json.loads(path.read_text())]
    out = cache / "all.json"
    out.write_text(json.dumps(merged, indent=1))
    print(f"{len(merged)} cards -> {out}; spent ${spent():.5f}")


if __name__ == "__main__":
    main()
