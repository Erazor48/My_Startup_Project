"""Simulated user who talks to the model over several days and reveals facts about themselves.

Statements (what the user says) and probes (what we ask at evaluation time) use disjoint
templates, so a correct answer requires remembering the fact, not reciting the dialogue.
"""

import random
from dataclasses import dataclass, field

FIRST_NAMES = [
    "Alice", "Bruno", "Chloe", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ines", "Jonas",
    "Kenji", "Lucia", "Marco", "Nadia", "Oscar", "Priya", "Quentin", "Rosa", "Samir", "Tilda",
    "Umberto", "Vera", "Wim", "Yara", "Zoltan",
]
CITIES = ["Nantes", "Lyon", "Lisbon", "Montreal", "Kyoto", "Oslo", "Seville", "Prague", "Dublin", "Tunis"]
JOBS = ["nurse", "carpenter", "librarian", "pilot", "baker", "architect", "plumber", "teacher", "chemist", "florist"]
COMPANIES = ["Airbus", "Decathlon", "Michelin", "Ubisoft", "Danone", "Renault", "Dassault", "Thales", "Orange", "Alstom"]

FAVORITES = {
    "dish": ["raclette", "ramen", "paella", "couscous", "lasagna", "pho", "tacos", "moussaka"],
    "color": ["turquoise", "crimson", "ochre", "lavender", "emerald", "mustard", "navy", "coral"],
    "sport": ["badminton", "rugby", "climbing", "fencing", "rowing", "judo", "handball", "surfing"],
    "drink": ["kombucha", "espresso", "lemonade", "matcha", "cider", "hot chocolate", "mint tea", "orangina"],
    "season": ["spring", "summer", "autumn", "winter"],
    "board game": ["chess", "Catan", "Carcassonne", "Dixit", "Risk", "Scrabble", "Azul", "Go"],
    "fruit": ["mango", "apricot", "lychee", "fig", "pomegranate", "kiwi", "cherry", "papaya"],
    "music genre": ["jazz", "reggae", "techno", "flamenco", "blues", "opera", "k-pop", "bossa nova"],
    "flower": ["tulip", "peony", "orchid", "sunflower", "lily", "daisy", "magnolia", "lilac"],
    "car brand": ["Peugeot", "Volvo", "Toyota", "Fiat", "Skoda", "Mazda", "Citroen", "Kia"],
    "movie genre": ["western", "horror", "documentary", "musical", "thriller", "animation", "sci-fi", "romance"],
    "dessert": ["tiramisu", "cheesecake", "baklava", "creme brulee", "mochi", "churros", "pavlova", "eclair"],
    "animal": ["otter", "owl", "fox", "dolphin", "panda", "lynx", "hedgehog", "penguin"],
    "instrument": ["cello", "banjo", "harp", "saxophone", "ukulele", "drums", "violin", "accordion"],
    "planet": ["Mars", "Jupiter", "Saturn", "Neptune", "Venus", "Mercury", "Uranus"],
    "language": ["Japanese", "Portuguese", "Swahili", "Finnish", "Arabic", "Korean", "Greek", "Hindi"],
}

RELATIONS = [
    "sister", "brother", "mother", "father", "best friend", "boss", "neighbor",
    "grandmother", "cousin", "dentist", "cat", "dog", "parrot", "landlord", "coach",
]

FILLER = [
    "Pretty tired today, long day.",
    "The weather is weird this week.",
    "I watched a documentary about volcanoes yesterday.",
    "Do you think it's going to rain tomorrow?",
    "I need to do my laundry tonight.",
    "My coffee went cold again, classic.",
    "Traffic was terrible this morning.",
    "I'm thinking about reorganizing my desk.",
    "Anyway, how are you doing?",
    "I forgot my umbrella again.",
    "The new season of that show is out, apparently.",
    "I should really drink more water.",
]

# Indirect probes share as few words as possible with the statements, to defeat similarity search.
INDIRECT = {
    "home_city": "If someone wanted to visit me at my place, which city would they travel to?",
    "job": "When people at a party ask what I do all day, what do I tell them?",
    "employer": "Whose logo is printed on my work badge?",
    "favorite_dish": "I'm cooking my comfort meal tonight. What's on the menu?",
    "favorite_color": "I'm repainting my bedroom walls in the shade I love. Which one?",
    "favorite_sport": "What activity should I sign up for to have fun this weekend?",
    "favorite_drink": "I'm at a cafe and want my usual treat to sip. What do I order?",
    "favorite_season": "What time of year do I look forward to the most?",
    "favorite_board_game": "For game night, which box should I bring from the shelf?",
    "favorite_fruit": "I'm making a smoothie with my top pick from the market stall. Which one?",
    "favorite_music_genre": "What kind of tunes should you play to cheer me up?",
    "favorite_flower": "What should someone bring me as a bouquet?",
    "favorite_car_brand": "If I bought a new vehicle tomorrow, which manufacturer would I pick?",
    "favorite_movie_genre": "For movie night, what kind of film should we put on for me?",
    "favorite_dessert": "It's my birthday, what sweet should the bakery prepare for me?",
    "favorite_animal": "Which creature should be on a plush toy gift for me?",
    "favorite_instrument": "If I took music lessons, what would I want to learn to play?",
    "favorite_planet": "Which world in our solar system fascinates me the most?",
    "favorite_language": "Which foreign tongue would I love to master?",
    "name_sister": "Who's my sis?",
    "name_brother": "Who's my bro?",
    "name_mother": "Who is my mom?",
    "name_father": "Who is my dad?",
    "name_best_friend": "Among all my pals, who do I trust the most?",
    "name_boss": "Who is my manager at work?",
    "name_neighbor": "Who lives next door to me?",
    "name_grandmother": "Who's my grandma?",
    "name_cousin": "Who is my aunt's kid?",
    "name_dentist": "Who should I call about my toothache?",
    "name_cat": "What's the name of my feline?",
    "name_dog": "What do I shout at the park when my pup runs off?",
    "name_parrot": "What's my talking bird called?",
    "name_landlord": "Who do I pay my rent to?",
    "name_coach": "Who runs my training sessions?",
}

# Chain facts are about a relative, named without the relation: answering needs two memories.
HUMAN_RELATIONS = [r for r in RELATIONS if r not in ("cat", "dog", "parrot")]
CHAINS = {
    "city": (CITIES, ("{n} just moved to {v}.", "{n} found a flat in {v}."), "Which city does my {rel} live in?"),
    "employer": (COMPANIES, ("{n} got hired at {v}.", "{n} started a new job at {v}."), "Which company does my {rel} work for?"),
}

ACKS = ["Got it!", "Nice, thanks for sharing.", "Noted.", "Interesting!", "Okay!", "Good to know."]


@dataclass(frozen=True)
class Slot:
    key: str
    pool: tuple[str, ...]
    statements: tuple[str, ...]
    probes: tuple[str, ...]
    updates: tuple[str, ...] = ()


def _build_slots() -> list[Slot]:
    slots = [
        Slot(
            "home_city", tuple(CITIES),
            ("I live in {v}.", "My apartment is in {v}, close to the river."),
            ("Which city am I living in?", "Where is my home located?"),
            ("Big news: I just moved to {v}!", "I relocated to {v} last week."),
        ),
        Slot(
            "job", tuple(JOBS),
            ("I work as a {v}.", "Being a {v} is my job, it's tiring but I like it."),
            ("What do I do for a living?", "What is my profession?"),
            ("I changed careers, I'm a {v} now.", "I quit my old job, I just started working as a {v}."),
        ),
        Slot(
            "employer", tuple(COMPANIES),
            ("I'm employed at {v}.", "My company is {v}."),
            ("Which company employs me?", "Who is my employer?"),
            ("I signed a contract with {v}, I start Monday.",),
        ),
    ]
    for cat, pool in FAVORITES.items():
        slots.append(Slot(
            f"favorite_{cat.replace(' ', '_')}", tuple(pool),
            (f"My favorite {cat} is {{v}}.", f"When it comes to {cat}, {{v}} is the best, no contest."),
            (f"Which {cat} do I like the most?", f"If you had to pick a {cat} for me, which one would I enjoy most?"),
        ))
    for rel in RELATIONS:
        slots.append(Slot(
            f"name_{rel.replace(' ', '_')}", tuple(FIRST_NAMES),
            (f"My {rel} is called {{v}}.", f"{{v}}, my {rel}, says hi by the way."),
            (f"What is my {rel}'s name?", f"How is my {rel} named?"),
        ))
    return slots


SLOTS = _build_slots()


@dataclass(frozen=True)
class Fact:
    slot: str
    value: str
    day: int
    statement: str
    replaces: str | None = None
    chain_question: str | None = None
    pool: tuple[str, ...] = ()


@dataclass(frozen=True)
class Probe:
    kind: str  # direct | indirect | chain
    slot: str
    question: str
    answer: str
    distractors: tuple[str, ...]
    stale: str | None


@dataclass
class World:
    seed: int
    days: list[list[tuple[str, str]]] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    probe_questions: dict[str, str] = field(default_factory=dict)

    def truth(self, day: int) -> dict[str, Fact]:
        state: dict[str, Fact] = {}
        for f in self.facts:
            if f.day <= day:
                state[f.slot] = f
        return state

    def probes(self, day: int) -> list[Probe]:
        """Questions about everything the user said up to `day` (inclusive), with current values."""
        assigned = {f.value for f in self.facts}
        out = []
        for slot, fact in self.truth(day).items():
            distractors = tuple(v for v in fact.pool if v != fact.value and v in assigned)
            if fact.chain_question:
                out.append(Probe("chain", slot, fact.chain_question, fact.value, distractors, None))
                continue
            for kind, question in (("direct", self.probe_questions[slot]), ("indirect", INDIRECT[slot])):
                out.append(Probe(kind, slot, question, fact.value, distractors, fact.replaces))
        return out


def make_world(
    n_days: int = 10, facts_per_day: int = 3, n_updates: int = 3, n_chains: int = 4, seed: int = 0
) -> World:
    rng = random.Random(seed)
    updatable_slots = [s for s in SLOTS if s.updates]
    others = [s for s in SLOTS if not s.updates]
    slots = rng.sample(others, n_days * facts_per_day - len(updatable_slots))
    # Updatable facts must first appear before the last day, so the update can happen later.
    for slot in updatable_slots:
        slots.insert(rng.randrange((n_days - 1) * facts_per_day), slot)
    world = World(seed=seed)

    value_of: dict[str, str] = {}
    first_day: dict[str, int] = {}
    schedule: dict[int, list[Fact]] = {d: [] for d in range(1, n_days + 1)}

    # Names, cities and companies are unique across the world so a probe has exactly one right answer.
    used: set[str] = set()

    def draw(pool: tuple[str, ...], unique: bool) -> str:
        value = rng.choice([v for v in pool if v not in used])
        if unique:
            used.add(value)
        return value

    for i, slot in enumerate(slots):
        day = i // facts_per_day + 1
        value = draw(slot.pool, unique=slot.key.startswith("name_") or bool(slot.updates))
        value_of[slot.key] = value
        first_day[slot.key] = day
        statement = rng.choice(slot.statements).format(v=value)
        schedule[day].append(Fact(slot.key, value, day, statement, pool=slot.pool))
        world.probe_questions[slot.key] = rng.choice(slot.probes)

    updatable = [s for s in slots if s.updates and first_day[s.key] < n_days]
    for slot in rng.sample(updatable, min(n_updates, len(updatable))):
        day = rng.randint(first_day[slot.key] + 1, n_days)
        old = value_of[slot.key]
        new = draw(slot.pool, unique=True)
        statement = rng.choice(slot.updates).format(v=new)
        schedule[day].append(Fact(slot.key, new, day, statement, replaces=old, pool=slot.pool))

    relatives = [
        s for s in slots
        if s.key.removeprefix("name_").replace("_", " ") in HUMAN_RELATIONS and first_day[s.key] < n_days
    ]
    for slot in rng.sample(relatives, min(n_chains, len(relatives))):
        rel = slot.key.removeprefix("name_").replace("_", " ")
        attr = rng.choice(sorted(CHAINS))
        pool, statements, question = CHAINS[attr]
        day = rng.randint(first_day[slot.key] + 1, n_days)
        value = draw(tuple(pool), unique=True)
        statement = rng.choice(statements).format(n=value_of[slot.key], v=value)
        schedule[day].append(Fact(
            f"chain_{slot.key}_{attr}", value, day, statement,
            chain_question=question.format(rel=rel), pool=tuple(pool),
        ))

    for day in range(1, n_days + 1):
        utterances = [f.statement for f in schedule[day]] + rng.sample(FILLER, 3)
        rng.shuffle(utterances)
        world.days.append([(u, rng.choice(ACKS)) for u in utterances])
        world.facts.extend(schedule[day])
    return world


if __name__ == "__main__":
    w = make_world()
    for d, turns in enumerate(w.days, 1):
        print(f"--- day {d}")
        for user, assistant in turns:
            print(f"  user: {user}\n  asst: {assistant}")
    print(f"\n{len(w.facts)} facts, {sum(f.replaces is not None for f in w.facts)} updates")
    for p in w.probes(len(w.days)):
        if p.kind != "direct":
            print(f"[{p.kind}] {p.question} -> {p.answer}")
