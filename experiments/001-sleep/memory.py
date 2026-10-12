"""Memory conditions compared in experiment 001. Each one lives through days, sleeps, and answers."""

import torch
from sentence_transformers import SentenceTransformer

from lm import LM, Reply

Turn = tuple[str, str]


class NoMemory:
    """C0 — floor: the model never sees the past."""

    name = "c0_none"

    def __init__(self, lm: LM):
        self.lm = lm

    def live_day(self, turns: list[Turn]) -> None:
        pass

    def sleep(self) -> None:
        pass

    def answer(self, question: str) -> Reply:
        return self.lm.chat([{"role": "user", "content": question}])


class FullContext(NoMemory):
    """C1 — ceiling: the whole history is replayed in the prompt at every question."""

    name = "c1_full_context"

    def __init__(self, lm: LM):
        super().__init__(lm)
        self.history: list[dict] = []

    def live_day(self, turns: list[Turn]) -> None:
        for user, assistant in turns:
            self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

    def answer(self, question: str) -> Reply:
        return self.lm.chat([*self.history, {"role": "user", "content": question}])


class Rag(NoMemory):
    """C2 — what products do today: embed past user messages, retrieve the closest ones per question.

    Snippets carry their day so the model can prefer the most recent value after an update.
    """

    name = "c2_rag"
    top_k = 4

    def __init__(self, lm: LM):
        super().__init__(lm)
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.snippets: list[str] = []
        self.embeddings = None
        self.day = 0

    def live_day(self, turns: list[Turn]) -> None:
        self.day += 1
        new = [f"(day {self.day}) {user}" for user, _ in turns]
        emb = self.encoder.encode(new, convert_to_tensor=True, normalize_embeddings=True)
        self.snippets += new
        self.embeddings = emb if self.embeddings is None else torch.cat([self.embeddings, emb])

    def retrieve(self, question: str) -> list[str]:
        q = self.encoder.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        top = (self.embeddings @ q).topk(min(self.top_k, len(self.snippets))).indices
        return [self.snippets[i] for i in sorted(top.tolist())]

    def answer(self, question: str) -> Reply:
        memories = "\n".join(self.retrieve(question))
        prompt = f"Things I told you in past conversations (most recent day wins):\n{memories}\n\n{question}"
        return self.lm.chat([{"role": "user", "content": prompt}])


CONDITIONS = {c.name: c for c in (NoMemory, FullContext, Rag)}
