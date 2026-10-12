"""Sleep conditions: memories are consolidated into LoRA weights at night, never put in the prompt.

Night = dream (the model writes Q/A pairs about the day's experiences) -> optional replay of past
dreams -> optional integration of new experiences with related old ones -> LoRA training.
"""

import random
import re

import torch
from peft import LoraConfig, get_peft_model

from lm import LM
from memory import NoMemory, Turn

DREAM_SYSTEM = "You are a memory system. Follow the format of the examples exactly."
# Few-shot examples are needed: a 2B model ignores abstract format instructions.
# They avoid the exact wording of the evaluation probes. The uncle example shows both directions: from
# "Samir, my cousin, says hi" the model only wrote "Who is Samir?", and a model trained on "Samir is your
# cousin" does not learn "your cousin is Samir" (the reversal curse).
DREAM_PROMPT = """Turn what the user said into flashcards. Questions are asked BY the user, in first person ("my", "I"). Answers address the user as "you". If there is nothing worth remembering about the user, write NOTHING.

Said: "My dog is called Rex."
Q: What did I name my dog?
A: Your dog is called Rex.
Q: Do I have a pet?
A: Yes, a dog called Rex.

Said: "Paul, my uncle, sent me a postcard."
Q: Who is my uncle?
A: Your uncle is Paul.
Q: Who is Paul?
A: Paul is your uncle.

Said: "The bus was late again."
NOTHING

Said: "{u}"
"""
INTEGRATE_PROMPT = """Two memories of the user mention the same person. Write flashcards about what follows when combining them (questions asked BY the user, first person).

Old: "My brother is called Tom."
New: "Tom got hired at Airbus."
Q: Where is my brother employed?
A: Your brother Tom works at Airbus.

Old: "Lea, my aunt, sends her love."
New: "Lea bought a house in Lille."
Q: In which city does my aunt live?
A: Your aunt Lea lives in Lille.

Old: "{old}"
New: "{new}"
"""
GENERAL_QUESTIONS = [
    "What is the capital of Italy?", "How many legs does a spider have?", "What is 12 times 7?",
    "Give me a synonym for 'happy'.", "What gas do plants absorb?", "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?", "Name a primary color.",
    "What is the largest ocean on Earth?", "How many days are in a leap year?",
    "Translate 'thank you' into Spanish.", "What is the square root of 81?",
    "Which planet is known as the Red Planet?", "What do bees make?",
    "What is the opposite of 'cold'?", "How many continents are there?",
]
QA = re.compile(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)\s*(?=\n|$)")
CAPITALIZED = re.compile(r"\b[A-Z][a-z]+\b")
NAME_IS_RELATION = re.compile(r"^([A-Z][a-z]+) is (?:your|my) ([a-z][a-z ]*?)\.?$")
NOT_ENTITIES = {"I", "My", "The", "Big", "When", "Honestly", "Anyway", "Do", "Pretty", "Traffic", "Being", "Today", "Earlier"}


STOPWORDS = {
    "the", "and", "you", "your", "are", "was", "were", "this", "that", "with", "for", "have", "has", "just",
    "called", "named", "name", "favorite", "best", "work", "works", "working", "live", "lives", "living",
    "moved", "who", "what", "which", "where", "now", "new", "old", "about", "like", "likes", "user",
}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z-]{2,}", text.lower()) if w not in STOPWORDS}


def grounded(cards: list[tuple[str, str]], *sources: str, ignore: set[str] = frozenset()) -> list[tuple[str, str]]:
    """Keep cards whose answer brings a content word of the source memories that the question lacks.

    Drops false memories copied from the prompt's few-shot examples (the dog Rex, the cousin Lea...).
    """
    vocab = set().union(*(words(s) for s in sources)) - ignore
    return [(q, a) for q, a in cards if (words(a) - words(q)) & vocab]


def mirrored(cards: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reverse "Samir is your cousin" into "Who is my cousin? -> Your cousin is Samir".

    The dream often keeps the direction of the sentence heard ("Samir, my cousin, says hi"), and training
    on one direction does not teach the other (the reversal curse).
    """
    matches = (NAME_IS_RELATION.match(a.strip()) for _, a in cards)
    return [(f"Who is my {rel}?", f"Your {rel} is {name}.") for name, rel in (m.groups() for m in matches if m)]


def first_answer(text: str) -> str:
    # The model sometimes keeps inventing new few-shot examples: keep only its first answer.
    return re.split(r"\n\s*(?:Said|Old):", text)[0]


def parse_cards(text: str) -> list[tuple[str, str]]:
    return QA.findall(first_answer(text) + "\n")


def entities(text: str) -> set[str]:
    return set(CAPITALIZED.findall(text)) - NOT_ENTITIES


class Sleep(NoMemory):
    """C3 — naive sleep: dream about today, train on it, nothing else."""

    name = "c3_sleep_naive"
    replay = False
    integrate = False
    replay_size = 24
    epochs = 3
    batch_size = 8
    lr = 2e-4

    def __init__(self, lm: LM, seed: int = 0):
        super().__init__(lm)
        self.rng = random.Random(seed)
        self.buffer: list[str] = []
        self.past: list[str] = []
        self.cards: dict[str, list[tuple[str, str]]] = {}  # source utterance -> its flashcards
        self.general: list[tuple[str, str]] = []
        lm.model = get_peft_model(lm.model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
        self.log: list[dict] = []

    def live_day(self, turns: list[Turn]) -> None:
        self.buffer += [user for user, _ in turns]

    def sleep(self) -> None:
        if self.replay and not self.general:
            # Rehearsal of general skills: the model's own answers, captured before any consolidation.
            with self.lm.model.disable_adapter():
                answers = self.lm.complete_batch(GENERAL_QUESTIONS, system="Answer in one short sentence.", max_new_tokens=32)
            self.general = list(zip(GENERAL_QUESTIONS, answers))

        today = self.dream(self.buffer)
        if self.integrate:
            today += self.integrate_with_past(self.buffer)

        examples = list(today)
        if self.replay:
            old = [c for u, cs in self.cards.items() if u not in self.buffer for c in cs]
            examples += self.rng.sample(old, min(self.replay_size, len(old)))
            examples += self.rng.sample(self.general, min(8, len(self.general)))

        loss = self.train(examples)
        self.log.append({"today_cards": len(today), "examples": len(examples), "loss": loss, "cards": today})
        self.past += self.buffer
        self.buffer = []

    def dream(self, utterances: list[str]) -> list[tuple[str, str]]:
        # The hippocampus dreams, not the cortex: generate with the base model, since the adapter
        # trained on short answers makes the model stop following the flashcard format.
        with self.lm.model.disable_adapter():
            outputs = self.lm.complete_batch([DREAM_PROMPT.format(u=u) for u in utterances], system=DREAM_SYSTEM)
        cards = []
        for u, out in zip(utterances, outputs):
            parsed = grounded(parse_cards(out), u)
            parsed += mirrored(parsed)
            self.cards[u] = parsed
            cards += parsed
        return cards

    def integrate_with_past(self, utterances: list[str]) -> list[tuple[str, str]]:
        # Reactivate the old memories that share a named entity: the new one elaborates them (Elena -> Elena
        # moved). Pairing by mere similarity bound one relative's name to another's relation ("your mother
        # is Samir"), and the 2B model's "outdated" verdicts dropped true memories from replay.
        pairs = [(p, u) for u in utterances for p in self.past if p != u and entities(u) & entities(p)]
        if not pairs:
            return []
        with self.lm.model.disable_adapter():
            outputs = self.lm.complete_batch(
                [INTEGRATE_PROMPT.format(old=p, new=u) for p, u in pairs], system=DREAM_SYSTEM, max_new_tokens=160
            )
        cards = []
        for (old, new), out in zip(pairs, outputs):
            # The shared name alone grounds nothing: "your boss Hugo is unemployed" invents the rest.
            shared = {e.lower() for e in entities(old) & entities(new)}
            parsed = grounded(parse_cards(out), old, new, ignore=shared)
            self.cards[new] = self.cards.get(new, []) + parsed
            cards += parsed
        return cards

    def train(self, examples: list[tuple[str, str]]) -> float | None:
        if not examples:
            return None
        tok = self.lm.tokenizer
        tok.padding_side = "right"
        model = self.lm.model
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        losses = []
        for _ in range(self.epochs):
            self.rng.shuffle(examples)
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i + self.batch_size]
                prompts = [self.lm.render([{"role": "user", "content": q}]) for q, _ in batch]
                fulls = [p + a + tok.eos_token for p, (_, a) in zip(prompts, batch)]
                enc = tok(fulls, return_tensors="pt", padding=True)
                labels = enc.input_ids.clone()
                labels[enc.attention_mask == 0] = -100
                for row, p in enumerate(prompts):
                    labels[row, :len(tok(p).input_ids)] = -100  # learn the answer, not the question
                loss = model(**enc, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
        model.eval()
        return sum(losses[-5:]) / min(5, len(losses))


class SleepReplay(Sleep):
    """C4 — sleep with replay of past dreams and of general skills (against catastrophic forgetting)."""

    name = "c4_sleep_replay"
    replay = True


class SleepIntegrate(SleepReplay):
    """C5 — replay + integration: new experiences reactivate the old ones about the same person and are combined."""

    name = "c5_sleep_integrate"
    integrate = True


CONDITIONS = {c.name: c for c in (Sleep, SleepReplay, SleepIntegrate)}
