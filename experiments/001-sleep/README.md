# Experiment 001 — Sleep: consolidating memories into the weights

Can a small LLM keep learning from what it lives through by **consolidating each day into its
weights at night**, instead of re-reading its past in the prompt like retrieval-augmented (RAG)
assistants do?

> Status: phase 0 closed. 3 seeds per condition, all numbers below come from `runs/` through
> `analyze.py`. One-line summary: sleep puts **no memory in the prompt** and may beat RAG on
> two-memory questions, but the evidence for that is 12 questions, and it recalls isolated facts
> less well than RAG. Read "Caveats" before quoting any number.

## Hypotheses

- **H1** — After 10 simulated days, a model that sleeps (LoRA trained on the day's memories plus
  replay of older ones) matches RAG on isolated facts and **beats it on questions that need two
  memories**, with **zero memory tokens in the prompt** and **< 2 points lost** on general skills.
- **H2** — Without replay, the model forgets earlier days (catastrophic forgetting).

## Setup

A simulated user talks to the model for 10 days. Each day they mention 3 facts about themselves
(relatives' names, job, city, employer, favorites) among small talk. Later days update some facts
("I relocated to Lyon last week") and give news about known people ("Samir got hired at
Danone"). One world = 30 facts + 3 updates + 4 news items, generated from templates with a seed.

After each night the model is asked about **everything said so far**, never with the wording of
the dialogue:

| Probe | Example | Tests |
|---|---|---|
| direct | "What is my sister's name?" | recall |
| indirect | "Who's my sis?" | recall when the question does not look like the memory |
| chain | "Which company does my cousin work for?" | combining two memories (cousin = Samir, Samir → Danone) |

Scoring is deterministic: the answer must contain the expected entity and no other entity of the
world from the same category.

## Conditions

| | Condition | Memory lives in | Role |
|---|---|---|---|
| C0 | no memory | — | floor |
| C1 | full history in the prompt | prompt | ceiling, maximal cost |
| C2 | RAG (MiniLM embeddings, top-4 past messages) | prompt | the baseline to beat |
| C3 | naive sleep: LoRA on today's memories | weights | shows forgetting (H2) |
| C4 | sleep + replay of past memories and general Q&A | weights | against forgetting |
| C5 | C4 + integration: new memories reactivate old ones about the same person and are combined | weights | our method |

A night, for C3-C5, with `Qwen/Qwen3.5-2B`:

1. **Dream**: the base model turns each of the day's messages into Q/A flashcards, in both
   directions ("Who is my cousin?" and "Who is Samir?"). Cards whose answer does not come from
   the message are dropped (see false memories below).
2. **Replay** (C4, C5): 24 cards from earlier nights and 8 general questions answered by the model
   before any training.
3. **Integration** (C5): each new message is paired with the old messages that share a name
   with it ("Dmitri, my grandmother, says hi" + "Dmitri found a flat in Nantes"), and the model
   writes cards about what follows from both.
4. **Consolidation**: LoRA (rank 16, all linear layers), 3 epochs. The day buffer is emptied.

The replayed cards are stored, like a RAG store, but they are only used for **training**, never
put in the prompt: inference cost stays constant however long the model has lived.

## Results

Mean of 3 seeds, day 10 (`uv run python analyze.py`; C0 was run on seed 0 only, its answers
are 0 % by construction). ARC-Easy: 200 fixed questions, "letter" = the model must answer with
the option letter, "cloze" = each option's text is scored by likelihood (no letters).

| Condition | direct | indirect | chain | old value after update | prompt tokens | ARC-Easy (letter) | ARC-Easy (cloze) |
|---|---|---|---|---|---|---|---|
| C0 no memory | 0 % | 0 % | 0 % | 0 % | 40 | 89.5 % | 72.5 % |
| C2 RAG | 93 % | 82 % | 17 % (2/12) | 22 % | 111 | 89.5 % | 72.5 % |
| C4 sleep + replay | 86 % | 60 % | 25 % (3/12) | 50 % | **41** | 84.3 % | 75.3 % |
| C5 sleep + integration | 81 % | 54 % | **75 % (9/12)** | 44 % | **41** | 79.3 % | 74.3 % |

C0 and C2 never change the weights, so their ARC columns are the base model's.

ARC-Easy per seed (200 questions, so one question = 0.5 point):

| Condition | letter, seed 0 / 1 / 2 | cloze, seed 0 / 1 / 2 |
|---|---|---|
| base model | 89.5 | 72.5 |
| C4 sleep + replay | 73.5 / 88.5 / 91.0 | 77.0 / 74.0 / 75.0 |
| C5 sleep + integration | 63.0 / 88.0 / 87.0 | 75.0 / 73.5 / 74.5 |

Chain questions per seed (4 per world): C5 gets 1/4, 4/4, 4/4; C4 gets 1/4, 1/4, 1/4; RAG gets
0/4, 0/4, 2/4.

Seed 0 alone (the first world, used while debugging):

| Condition | direct | indirect | chain | old value after update | prompt tokens | ARC-Easy (letter) |
|---|---|---|---|---|---|---|
| C0 no memory | 0 % | 0 % | 0 % | 0 % | 40 | 89.5 % |
| C2 RAG | 97 % | 77 % | 0 % | 17 % | 112 | 89.5 % |
| C4 sleep + replay | 87 % | 47 % | 25 % | 67 % | 40 | 73.5 % |
| C5 sleep + integration | 80 % | 63 % | 25 % | 67 % | 40 | 63.0 % |

- RAG retrieves isolated facts almost perfectly but answers no chain question: no single
  retrieved message contains the answer, and similarity search brings back the user's own
  employer instead of the cousin's.
- Sleep keeps memory out of the prompt (40 tokens = system prompt + question, like no memory at
  all) and recalls most direct facts, but loses on indirect questions and still gives the old
  value of updated facts (both values are replayed).
- C5 writes the right combined cards ("your cousin Samir works at Danone", "your best friend
  Bruno works at Thales") but mixes them up at test time (cousin -> Thales): two facts of the same
  shape, learned the same night from a single card each.
- Sleep costs 5 points (C4) to 10 points (C5) on average on ARC-Easy answered with a letter, but
  nothing when scored as cloze: the model unlearns the multiple-choice format, not the
  knowledge (see failure 6 and Caveats). H1 fails on this criterion for now.

## Caveats

- **The C5 chain result (75 %) rests on 12 questions** (9 correct), from 3 worlds of 4 chain
  questions each. It is carried by two seeds with 4/4; on seed 0, C5 gets 1/4, like C4. The RAG
  baseline gets 2/12. A confidence interval on 9/12 is roughly 47-91 % (Wilson, 95 %). This is a
  signal worth following up, not a measurement, and it is not enough to claim C5 beats RAG.
- **The general-skills loss on ARC-Easy is concentrated on one seed, and we do not know why.**
  With the answer given as a letter, seed 0 loses 16 points (C4) and 26.5 points (C5), while
  seeds 1 and 2 change by +1.5 to -2.5 points. We checked that it is not evaluation noise: the
  seed 0 adapter scores 73.5 % again when re-evaluated (the evaluation is deterministic), and
  the training code (`sleep.py`, `lm.py`) was last modified before the first adapter was trained. The seed 0 runs show a lower training loss on the first night (0.41 against
  0.63-0.65), but we have not shown that this is the cause, and a full re-run of seed 0 could not
  be finished on the machine. **The gap between seed 0 and the other seeds remains unexplained.**
  What is established: the loss is in the letter format only (cloze accuracy is intact on all
  seeds), and whether a given world damages that format is not predictable from 3 seeds.
- H1's criterion (< 2 points lost on general skills) is met in cloze (no loss on any seed). On
  the letter format it is met only by C4 seeds 1 and 2 and C5 seed 1.

## What went wrong, and what it taught us

Failures are part of the result.

1. **Generation did not stop at the end of a turn**: the model went on writing the user's next
   messages. Fixed by stopping on `<|im_end|>`.
2. **Dreaming with the trained model**: once the adapter learned short answers, the model no
   longer followed the flashcard format. Dreams are now produced by the base model, the way the
   hippocampus, not the cortex, replays the day.
3. **False memories**: the model copied the prompt's example ("my dog is called Rex") into its
   own memories, then learned it. Every card must now be grounded in what was actually said.
4. **The integration step created its own interference.** In the first C5 run, favorites learned
   on day 1 were still right on day 10, but relatives' names were learned and then overwritten by
   the latest name learned (sister: Kenji → Dmitri → Tilda → Chloe). Three causes, none of them
   in the LoRA itself:
   - pairing memories by embedding similarity made the model write cross-bound cards
     ("your mother is Samir");
   - the "is the old memory outdated?" verdict was read from the whole generation, including
     few-shot examples the model kept inventing;
   - even when read correctly, the 2B model's outdated verdicts removed 4 true facts (and 2
     small-talk messages) from replay for 3 real updates, including the user's *current* job and
     employer. A logit-scored judge did no better on held-out worlds: 2 of 5 updates found, 4 true
     facts removed ("I signed with Orange" removed the favorite fruit).

   C5 now combines only memories that share a named entity and never drops anything from
   replay. Retiring outdated memories is a problem of active forgetting, left to experiment 002.
5. **The reversal curse, in the dream.** Without integration (C4), 4 of 13 names were still
   wrong, all stated as "Samir, my cousin, says hi". From such sentences the model only wrote
   "Who is Samir? -> Your cousin", and a model trained on "A is B" does not learn "B is A"
   ([Berglund et al., 2023](https://arxiv.org/abs/2309.12288)). The 3 names with no
   relation -> name card were all wrong on day 10; 9 of the 10 others were right. Dreams now
   include both directions (a two-way few-shot example, and every "X is your Y" card gets its
   mirror).
6. **Sleep costs answer formats, not knowledge.** After 10 nights on seed 0, C4 fell from 89.5 % to
   73.5 % on ARC-Easy answered with a letter (34 questions lost, 2 gained), far from the
   < 2 points of H1 (seeds 1 and 2 did not fall, see Caveats). Scored as cloze (likelihood of
   each option's text, no letters, no chat template), the base model gets 72.5 % and the three
   C4 adapters 74.0-77.0 %: the knowledge is intact, the multiple-choice format is what was
   unlearned. The general replay only rehearses one format
   (one short sentence).

## Limitations

- Small and synthetic: 30 template facts per world, one 2B model, 3 seeds.
- Related memories are found by shared capitalized names, a heuristic that fits this world.
- Updated facts are replayed with both their old and new values in C4 and C5.
- If C5 beats RAG on chains, a fair follow-up is RAG + nightly reflection notes (as in
  *Generative Agents*), to tell whether the gain comes from the weights or just from the
  deduction.

## Reproduce

Everything runs on a laptop CPU (Ryzen AI 9 365, 30 GB RAM, no GPU). A full run of one sleeping
condition takes 15-50 minutes of sleep plus evaluation; the whole batch takes several hours.

The raw results of the runs reported above are committed in `runs/` (one JSON per condition and
seed, `runs/arc/` and `runs/arc_cloze/` for general skills). Adapters and logs are not.

```bash
uv sync
cd experiments/001-sleep
uv run python analyze.py                                                # tables from the committed runs, no model needed
uv run python run.py c4_sleep_replay --seed 0 --eval-days 1 3 5 7 10  # one condition, one world
uv run python arc.py adapters/c4_sleep_replay_seed0                     # general skills, letter
uv run python arc_cloze.py adapters/c4_sleep_replay_seed0               # general skills, cloze
./run_all.sh                                                            # everything, resumable
```

What is and is not checked:

- `analyze.py` regenerates every table above from the committed JSON files.
- Re-evaluating a given adapter on ARC-Easy gives the same score (checked on C4 seed 0: 73.5 %
  twice).
- Re-training an adapter from scratch is **not** shown to give the same numbers: a re-run of C4
  seed 0 was started and not finished. The seed fixes the world and the replay choices, but
  the LoRA initialization is not seeded (no `torch.manual_seed`) and CPU threading changes
  floating-point sums, so expect different sleep results for the same seed. Given the
  seed-to-seed spread on ARC-Easy, this may matter.
