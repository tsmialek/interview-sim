# Vocare Interview Simulator

Speech analysis and candidate evaluation module for a job interview simulator: recording, transcription and assessment of the candidate's answers (speech fluency, professionalism, style and semantics).

**Authors: Tomasz Śmiałek, Bartłomiej Kłuś, Kacper Jóźwik**

_Project description: [docs/NLP - Opis projektu.pdf](docs/NLP%20-%20Opis%20projektu.pdf)_

## Setup

To start working with the project you need conda installed. Create the environment and register the Jupyter kernel:

```bash
# create the environment
conda env create -f environment.yml

# activate it
conda activate interview-sim

# register the kernel (for Jupyter Notebook)
python -m ipykernel install --user --name interview-sim
```

The Polish `spaCy` model (`pl_core_news_md`) is installed together with the environment.

After `environment.yml` changes, update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

Docelowa struktura do nagrywania danych:

data/
├── questions.csv # lista pytań
├── recordings.csv # co jest w którym nagraniu
├── audio/ # nagrania
│ ├── r001.wav
│ └── r002.wav
└── transcripts/ # ręczne transkrypcje wzorcowe
├── r001.txt
└── r002.txt

# Recording Guidelines

The recordings dataset is used to test transcription, correction of domain-specific terms, and the assessment of style, fluency and content. Questions are in `data/questions.csv`, recording metadata in `data/recordings.csv`.

Recordings are in **Polish**, so the examples of colloquial words and fillers below are given in Polish.

## 1. Equipment and conditions

- Do **not** use filters, noise reduction or "voice enhancement" in the recording app.
- Target format: **WAV, 16 kHz, mono**. Convert phone recordings (`.m4a`) with:

  ```bash
  ffmpeg -i recording.m4a -ar 16000 -ac 1 q09_B_kacper.wav
  ```

## 2. How much to record

### Stage 1 (to get started, ~16 recordings per person)

```text
┌──────────────────────┬──────────┬───────┐
│ Question             │ Variants │ Count │
├──────────────────────┼──────────┼───────┤
│ q01 (about yourself) │ A, B     │   2   │
│ q02 (project)        │ A, B     │   2   │
│ q07 (Scrum)          │ A, B, C  │   3   │
│ q09 (REST API)       │ A, B, C  │   3   │
│ q11 (Docker)         │ A, B, C  │   3   │
│ q15 (PoC)            │ A, B, C  │   3   │
├──────────────────────┼──────────┼───────┤
│ Total                │          │   16  │
└──────────────────────┴──────────┴───────┘
```

This is enough to build and test the whole pipeline.

### Stage 2 (full dataset, ~54 recordings per person)

```text
┌──────────────────────────────────┬──────────┬───────┐
│ Questions                        │ Variants │ Count │
├──────────────────────────────────┼──────────┼───────┤
│ q01–q06 (general and behavioral) │ A, B     │   12  │
│ q07–q20 (process and technical)  │ A, B, C  │   42  │
│ optional: selected technical     │ D        │  any  │
├──────────────────────────────────┼──────────┼───────┤
│ Total                            │          │  54+  │
└──────────────────────────────────┴──────────┴───────┘
```

**Number of speakers:** the more different voices, the better. The minimum goal is 2–3 people, at least for Stage 1.

### Length of one answer

**45–120 seconds**, ideally about 60–90 s. Shorter recordings give unreliable statistics (speech rate, TTR).

## 3. Variants

```text
┌─────────┬────────────┬───────────┬─────────────────────────────────┐
│ Variant │ Style      │ Fluency   │ Content                         │
├─────────┼────────────┼───────────┼─────────────────────────────────┤
│ A       │ formal     │ fluent    │ correct                         │
│ B       │ colloquial │ disfluent │ correct                         │
│ C       │ formal     │ fluent    │ partial / incorrect / off-topic │
│ D       │ colloquial │ disfluent │ partial / incorrect             │
└─────────┴────────────┴───────────┴─────────────────────────────────┘
```

Questions q01–q06 have no single correct answer, so record only A and B for them (they differ in form only).

### A: model candidate

- Full sentences, professional vocabulary, a well-structured answer.
- Naturally use a few terms from the `expected_terms` column of the question.
- An occasional natural hesitation is fine, you don't have to sound like a narrator.

### B: knows the answer, but speaks poorly

- Correct content, but casual language: „spoko”, „ogarnąć”, „no i generalnie”, „jakby”, „ten tego”, „w sumie”.
- Fillers („yyy”, „eee”, „no”), longer pauses, repetitions („to jest… to jest takie”), broken-off and restarted sentences.
- Pronounce technical terms the Polish way, as in casual speech (see section 4).

### C: sounds good, says little

- Sounds professional and fluent, but the content is weak. Pick one:
  - **vague**: many words, few specifics („to bardzo ważna technologia, która usprawnia procesy”),
  - **with a mistake**: e.g. confusing authentication with authorization, or Docker with a virtual machine,
  - **off-topic**: start with the question and drift to a different subject.
- Write the intended mistake or drift in the `notes` column.

### D: worst case (optional)

A combination of B and C: colloquial, hesitant and weak on content.

### General rules

- **Don't read from a script.** Read text sounds unnatural and has different hesitations than spontaneous speech. You can keep 3–4 keywords in front of you.
- **Don't overact.** Slightly exaggerated is fine, a caricature is not.
- If a recording turned out different than planned, **don't redo it**. Keep it and describe what actually happened in the `label_*` columns. Such cases are valuable too.

## 4. Terms to include (especially in variants B and D)

Deliberately pronounce them "the Polish way" to test transcription correction:

```text
┌──────────┬────────────────────────────────────────┬────────────────────────────────────┐
│ Question │ Colloquial pronunciation               │ Correct form                       │
├──────────┼────────────────────────────────────────┼────────────────────────────────────┤
│ q07      │ dżira, skram, dejli, bakklog           │ Jira, Scrum, daily, backlog        │
│ q08      │ pul rekłest, kołd riwju, merdżować     │ pull request, code review, merge   │
│ q09      │ rest-api, dżej-son, endpojnt           │ REST API, JSON, endpoint           │
│ q11      │ doker, dokerfajl, doker kompołz        │ Docker, Dockerfile, docker-compose │
│ q12      │ ci-aj-si-di, gitchab ekszyns, dżenkins │ CI/CD, GitHub Actions, Jenkins     │
│ q13      │ gita, brancz, rebejs                   │ Git, branch, rebase                │
│ q15      │ pe-o-ka, POK, pruf of koncept          │ PoC, proof of concept              │
│ q18      │ dżej-dabl-ju-ti, o-aut                 │ JWT, OAuth                         │
│ q19      │ kubernetis, ku-ejts, jamel             │ Kubernetes, K8s, YAML              │
└──────────┴────────────────────────────────────────┴────────────────────────────────────┘
```

## 5. File names

```
<question_id>_<variant>_<speaker>.wav
```

Examples: `q09_B_kacper.wav`, `q15_C_ania.wav`. If you record the same variant a second time: `q09_B_kacper_2.wav`.

Files go to `data/audio/`.

## 6. Filling in `recordings.csv`

**Right after recording**, fill in: `file`, `question_id`, `speaker`, `variant`, `intended_style`, `intended_fluency`, `intended_content`, `duration_s`, `recorded_at`, `notes`.

**After listening** (ideally by another person, or at least the next day), fill in `label_style`, `label_fluency`, `label_content`. These are the ground-truth labels for precision/recall/F1. They describe what can actually be heard, not what was planned.

The CSV values stay in Polish. Allowed values:

```text
┌───────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Column    │ Values                                                                                                   │
├───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ *_style   │ formalny (formal), potoczny (colloquial)                                                                 │
│ *_fluency │ płynny (fluent), przerywany (disfluent)                                                                  │
│ *_content │ poprawna (correct), częściowa (partial), błędna (incorrect), nie_na_temat (off-topic); empty for q01–q06 │
└───────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Example:

```csv
q09_B_kacper.wav,q09,kacper,B,potoczny,przerywany,poprawna,potoczny,przerywany,poprawna,74,2026-11-01,"mówiłem ""dżej-son"" zamiast JSON"
```

## 7. Reference transcripts

For each recording, create a text file with the same name in `data/transcripts/`, e.g. `q09_B_kacper.txt`. Whisper's output is compared against these files (CER).

Writing rules:

- Write down **exactly what was said**, including repetitions, broken-off words and colloquialisms.
- Always write fillers the same way: `yyy`, `eee`, `mmm`. Write „no”, „jakby”, „w sumie” as regular words.
- **Write technical terms in their correct form** (`Jira`, `PoC`, `JSON`), even if they were pronounced colloquially, because that is the target of the correction.
- Write numbers as words, the way they were spoken.
- No pause markers. Pauses are computed from timestamps.
