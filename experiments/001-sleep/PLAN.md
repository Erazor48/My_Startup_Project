# Expérience 001 — « Sommeil » : consolider la mémoire dans les poids

Statut : **C0/C2/C4/C5 relancés avec le code corrigé, 3 seeds (batch de nuit, 2026-10-08)** · Durée visée : 4 semaines · Machine : Ryzen AI 9 365, 31 Go RAM, CPU only

## Question

Un petit LLM peut-il **apprendre en continu ce qu'il vit**, en consolidant chaque nuit ses
souvenirs dans ses poids, au lieu de les relire dans son contexte comme ChatGPT ?

## Hypothèse (falsifiable)

**H1** — Avec une phase de sommeil (LoRA entraîné sur les souvenirs du jour + rejeu des
souvenirs anciens), après 10 jours simulés, le modèle :
- retrouve **≥ 80 % de ce que retrouve un RAG**,
- avec **zéro token de mémoire dans le prompt** (coût d'inférence constant, quel que soit le
  nombre de jours vécus),
- en perdant **< 2 points** de capacité générale.

**H2** — Sans rejeu, le modèle oublie les jours anciens (oubli catastrophique). C'est le
mécanisme biologique qu'on veut démontrer utile.

## Le monde simulé

Un utilisateur fictif parle au modèle pendant **10 jours**. Chaque jour, il révèle 3 faits
sur lui (noms de proches, métier, ville, employeur, goûts), noyés dans de la conversation
banale. Total : 30 faits + 3 mises à jour. Réduit de 100 à 30 pour tenir sur CPU ; on
augmentera si les résultats sont prometteurs.

- **Faits qui changent** : la ville, le métier et l'employeur sont mis à jour plus tard (« j'ai déménagé à Lyon »
  au jour 7 remplace « je vis à Nantes » du jour 2). On teste la mise à jour, pas seulement le
  stockage.
- **Questions d'évaluation reformulées** : on ne pose jamais la question avec les mots du
  dialogue. Sinon on mesure de la récitation, pas de la mémoire.
- Génération **par templates + aléatoire avec seed** : déterministe, reproductible, gratuit.
- Langue : **anglais** (le 0.5B est incohérent en français).

## Conditions comparées

| # | Condition | Rôle |
|---|---|---|
| C0 | Pas de mémoire | plancher (≈ ce qu'on a vu dans Ollama) |
| C1 | Tout l'historique dans le prompt | plafond de rappel, coût maximal |
| C2 | RAG (embeddings + top-k des passages passés) | **baseline à battre**, c'est l'état de l'art produit |
| C3 | Sommeil naïf : LoRA sur les faits du jour seulement | doit montrer l'oubli (H2) |
| C4 | **Sommeil avec rejeu** : faits du jour + échantillon des jours passés + un peu de texte général | contre l'oubli |
| C5 | **Sommeil avec intégration** : chaque nouveau souvenir réactive les anciens qui parlent de la même personne ; le modèle en déduit des faits combinés | **notre méthode** |

L'oubli actif (élaguer ce qui ne sert pas, retirer ce qui est devenu faux) sera l'expérience 002. On isole un seul mécanisme
à la fois.

## Le sommeil, concrètement (C3/C4)

1. **Jour** : les dialogues vont dans un tampon (l'« hippocampe »).
2. **Nuit** :
   - extraction des faits du tampon ;
   - **augmentation** : chaque fait est réécrit en plusieurs paraphrases et paires
     question/réponse. Un LLM retient très mal un fait vu une seule fois (*Physics of
     Language Models 3.1*, Allen-Zhu & Li) ;
   - (C4) on ajoute un rejeu de faits des nuits précédentes et du texte général ;
   - entraînement d'un LoRA sur ce mélange ;
   - le tampon est vidé.
3. Le lendemain, le modèle répond **sans aucun souvenir dans son prompt**.

## Mesures

- **Rappel** : après chaque nuit, on interroge **tous** les faits vus jusque-là. On obtient une
  matrice jour × fait, donc une **courbe d'oubli**. Score automatique : la réponse contient-elle
  l'entité attendue (pas de juge LLM, résultat déterministe).
- **Mise à jour** : pour les faits changés, le modèle donne-t-il la nouvelle valeur ?
- **Capacité générale** : 200 questions ARC-Easy, avant et après les 10 nuits.
- **Coût** : tokens de prompt par question (inférence), durée d'entraînement par nuit.

## Critères de décision

- ✅ **Succès** : H1 tenue → on passe au village d'agents (étape 2 de la roadmap).
- 🔁 **Pivot** : C4 apprend les faits du jour même mais oublie malgré le rejeu → on teste un
  modèle plus gros (Qwen 1.5B sur Kaggle) ou une mémoire dédiée (type Titans).
- ❌ **Échec** : C4 < 50 % de rappel même sur les faits de la veille, malgré le réglage → la
  consolidation par LoRA est la mauvaise brique. On le documente, c'est un résultat aussi.

## Stack

- Python 3.12 + **uv**, venv local au projet.
- `torch` (CPU), `transformers`, `peft`, `sentence-transformers` (pour le RAG).
- Modèle : `Qwen/Qwen3.5-2B` (choix d'Ethan). Nativement multimodal (texte + image) : la vision
  est déjà là pour la suite. Architecture hybride (Gated DeltaNet + attention), non-thinking par défaut.
  Tout tourne dans `transformers` pour que les 5 conditions soient comparées dans les mêmes
  conditions. Ollama revient plus tard pour la démo.
- Mesuré sur le portable (CPU) : génération 3,6 tok/s, lecture de 2 000 tokens 18 s, un pas LoRA
  (batch 8) ≈ 6,5 s. Une nuit ≈ 10 min → tout tient en local.

## Arborescence

```
experiments/001-sleep/
  PLAN.md
  world.py      # génère les 10 jours, les faits, les questions reformulées
  memory.py     # C0..C4 : une interface commune answer(question)
  sleep.py      # augmentation + rejeu + entraînement LoRA
  evaluate.py   # matrice de rappel, mises à jour, ARC-Easy, coûts
  run.py        # boucle 10 jours pour une condition donnée
  arc.py        # ARC-Easy (capacité générale), modèle de base ou avec un adaptateur
  run_all.sh    # batch de nuit, reprenable
  results/      # JSON + graphiques
```

## Calendrier

| Semaine | Livrable |
|---|---|
| 1 | `world.py` + `evaluate.py` + C0 et C1 qui tournent → premiers chiffres |
| 2 | C2 (RAG) |
| 3 | C3 et C4 (sommeil) |
| 4 | Analyse, courbes d'oubli, README publiable sur GitHub (open source) |

## Constat semaine 1 : le RAG est très fort sur des faits isolés (à discuter)

Test de récupération seule : au jour 10, le RAG remonte le bon souvenir pour **30/30**
questions. Son coût d'inférence est aussi presque constant (4 extraits ≈ 80 tokens). Sur des
faits simples et isolés, le sommeil ne pourra donc au mieux qu'**égaler** le RAG. Pas de 10x.

Là où un cerveau (et donc le sommeil) devrait gagner, c'est sur **l'intégration** :
- **combiner** plusieurs souvenirs qu'aucun extrait ne contient seul (« ma sœur vient dîner,
  propose un dessert que j'aimerais tous les deux ») ;
- **inférer** ce qui n'a jamais été dit en une phrase (des goûts déduits de plusieurs jours) ;
- répondre quand la question ne **ressemble pas** au souvenir (le RAG cherche par similarité).

**Décidé (2026-10-05)** : le monde contient maintenant des questions **directes**, **indirectes**
(peu de mots communs avec le souvenir) et **en chaîne** (deux souvenirs à relier). Récupération
du RAG sur 5 mondes : 99 % / 91 % / **25 %**. H1 devient « le sommeil égale le RAG sur les faits
isolés et le **dépasse sur les questions en chaîne** ».

## Pourquoi C5 (intégration)

Le cerveau ne se contente pas de rejouer : pendant le sommeil, un nouveau souvenir réactive les
anciens qui lui ressemblent, et les deux sont réorganisés ensemble (théorie des schémas,
reconsolidation). C5 imite ça : « Elena a déménagé à Lisbonne » réactive « ma sœur s'appelle
Elena », et le modèle s'entraîne sur le fait déduit « ma sœur vit à Lisbonne ».

Limite honnête : un RAG peut faire la même chose (écrire des notes déduites la nuit, comme la
« réflexion » de *Generative Agents*). Si C5 bat C2, il faudra tester un **C2b = RAG + réflexion
nocturne** pour savoir si le gain vient du sommeil dans les poids ou juste de la déduction.

## Constat 2026-10-08 : C5 fabriquait lui-même son interférence

Au jour 10 du premier run de C5 (seed 0), les **goûts** (couleur, fruit, langue) appris aux jours
1-3 étaient tous encore justes. Les **prénoms de proches**, eux, étaient appris puis écrasés par le
dernier prénom appris (sœur : Kenji → Dmitri → Tilda → Chloé). Trois causes trouvées, toutes
dans l'étape d'intégration, pas dans le LoRA :

1. **Bug** : le verdict « OUTDATED: yes » était cherché dans toute la génération, y compris
   les faux exemples que le 2B continue d'inventer (il recopie « baker → pilot »).
2. **Fausses liaisons** : associer deux proches par simple similarité fait écrire au 2B des
   cartes croisées (« ta mère est Samir »), que le filtre `grounded()` laissait passer.
3. **Le 2B juge mal ce qui est périmé** : même sans le bug, 4 faits vrais (et 2 phrases de
   bavardage) retirés du rejeu pour 3 vraies mises à jour, dont le métier et l'employeur
   **actuels**. Un juge noté par les logits (porte oui/non, puis QCM) ne fait pas mieux sur les
   mondes de dev 10-11 : 2 bonnes suppressions (sur 5 mises à jour) pour 4 faits vrais retirés
   (« contrat chez Orange » → retire le fruit préféré).

**Changement (à valider par Ethan)** : C5 ne combine plus que les souvenirs qui partagent une
entité nommée (« Dmitri, ma grand-mère » + « Dmitri a trouvé un appart à Nantes »), et ne retire
plus rien du rejeu. Retirer les faits périmés est un problème d'**oubli actif** : il part dans
l'expérience 002. Conséquence : C4 et C5 rejouent encore l'ancienne valeur après une mise à
jour ; la colonne « stale » le mesurera honnêtement.

Premiers chiffres avec le code corrigé (seed 0, jour 10) : C2 (RAG) directes 29/30, indirectes
23/30, **chaîne 0/4**. ARC-Easy du modèle de base : **89,5 %** (179/200, score par les logits).

**C4 seed 0 (sans intégration)** : directes 25/30, indirectes 16/30, chaîne 0/4. La sœur, le
propriétaire et le métier restent justes du jour 3 au jour 10 : l'écrasement venait bien de
l'intégration. Mais 4 prénoms sur 13 restent faux, et ce n'est pas du LoRA : c'est le **rêve**.
De « Samir, my cousin, says hi », le 2B n'écrit que « Who is Samir? → Your cousin ». Or un modèle
entraîné sur « A est B » n'apprend pas « B est A » (*Reversal Curse*, Berglund et al. 2023). Les 3
prénoms sans aucune carte « relation → prénom » sont tous faux au jour 10 ; 9 sur 10 des autres
sont justes. Correctif : un exemple à deux sens dans le prompt de rêve, et chaque carte
« X is your Y » reçoit sa carte miroir (entraînement inversé). Sur les 39 prénoms des mondes 0-2,
il manque maintenant 0 carte « relation → prénom » (4 avec le prompt seul). C4 et C5 sont relancés.

**Capacités générales : H1 échoue nettement.** Après 10 nuits, C4 passe de 89,5 % à **74,0 %** sur
ARC-Easy (34 questions perdues, 3 gagnées : ce n'est pas du bruit). Avec le nouveau rêve : 73,5 %
(34 perdues, 2 gagnées), donc le rêve n'y est pour rien. Le critère était < 2 points. Le
rejeu de 8 questions générales par nuit ne protège pas.

**Mais c'est le format qui se perd, pas le savoir.** Noté en *cloze* (probabilité du texte de
chaque réponse après « Question: … Answer: », sans lettres ni chat), le modèle de base fait
72,5 % et les modèles après 10 nuits 73,5-77,0 % (3 seeds, `arc_cloze.py` ; l'ancienne mesure 66,5 / 71,5 % venait d'un script jetable et n'est pas reproductible). Le savoir est intact ; ce qui se perd, c'est la
capacité à répondre à un QCM par une lettre. Le rejeu général n'entraîne qu'un seul format
(« une phrase courte ») : le modèle désapprend les autres. Piste : un rejeu général aux formats
variés (QCM, listes, réponses longues), produits par le modèle de base lui-même.

## Risques connus

- **Le LoRA peut mal retenir des faits vus peu de fois.** On le saura en semaine 3. Repli :
  plus de paraphrases, rang LoRA plus élevé, ou une mémoire dédiée (type Titans).
- **Faux succès par récitation** : évité par les questions reformulées.
- **Rejeu = tricherie ?** Le rejeu stocke des faits passés quelque part, comme un RAG. La
  différence défendable : ils servent à **entraîner**, jamais à **répondre**. Le coût
  d'inférence reste constant. À formuler clairement dans le README.
