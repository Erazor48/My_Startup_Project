"""LoCoMo (Maharana et al., 2024) adapted to the sleep loop: one session = one day, then one night.

The official F1 is re-implemented from snap-research/locomo (task_eval/evaluation.py): Porter-stemmed
token F1, comma-separated sub-answers for multi-hop, first ';' part for open-domain. Adversarial
questions (category 5) are left out, as Mem0 and Zep do, which leaves the 1,540 questions they report.
"""

import json
import re
import string
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.stem import PorterStemmer

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA = Path(__file__).parent / "data" / "locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
STEM = PorterStemmer()


@dataclass
class Session:
    date: str
    turns: list[tuple[str, str]]  # (speaker, text)


@dataclass
class Question:
    category: int
    question: str
    answer: str
    sessions: frozenset[int] = frozenset()  # sessions holding the evidence

    def answerable_after(self, n: int) -> bool:
        return bool(self.sessions) and max(self.sessions) <= n


@dataclass
class Conversation:
    sample_id: str
    sessions: list[Session]
    questions: list[Question]


def load(index: int) -> Conversation:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    raw = json.loads(DATA.read_text())[index]
    conv = raw["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    sessions = [
        Session(conv[f"{k}_date_time"], [(t["speaker"], t["text"]) for t in conv[k]]) for k in keys
    ]
    questions = [
        Question(q["category"], q["question"], str(q["answer"]), frozenset(int(n) for n in re.findall(r"D(\d+):", " ".join(q["evidence"]))))
        for q in raw["qa"] if q["category"] in CATEGORIES
    ]
    return Conversation(raw["sample_id"], sessions, questions)


def date_only(date_time: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '8 May 2023'."""
    return date_time.split(" on ", 1)[-1].replace(",", "")


def normalize(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, truth: str) -> float:
    p = [STEM.stem(w) for w in normalize(prediction).split()]
    t = [STEM.stem(w) for w in normalize(truth).split()]
    same = sum((Counter(p) & Counter(t)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def question_f1(q: Question, prediction: str) -> float:
    truth = q.answer
    if q.category == 3:
        truth = truth.split(";")[0].strip()
    if q.category == 1:
        predictions = [p.strip() for p in prediction.split(",")]
        truths = [g.strip() for g in truth.split(",")]
        return sum(max(f1_score(p, g) for p in predictions) for g in truths) / len(truths)
    return f1_score(prediction, truth)
