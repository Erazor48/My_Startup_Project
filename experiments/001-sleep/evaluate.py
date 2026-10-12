import re

from world import Probe


def _mentions(text: str, value: str) -> bool:
    return re.search(rf"\b{re.escape(value)}(?:e?s)?\b", text, re.IGNORECASE) is not None


def score(answer: str, probe: Probe) -> str:
    """'correct' | 'stale' (gives the value before an update) | 'wrong'.

    Naming a distractor alongside the right value counts as wrong: listing candidates is not remembering.
    """
    if _mentions(answer, probe.answer) and not any(_mentions(answer, d) for d in probe.distractors):
        return "correct"
    if probe.stale and _mentions(answer, probe.stale):
        return "stale"
    return "wrong"
