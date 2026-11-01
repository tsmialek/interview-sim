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

Na ten moment stworzona została lista a folderami i plikami (dodałem 20 przykładowych pytań)

Instrukcja nagrywania odpowiedzi

Zbiór nagrań służy do testowania transkrypcji, korekty terminów branżowych oraz oceny stylu, płynności i merytoryczności. Pytania są w data/questions.csv, opis nagrań w data/recordings.csv.

1. Sprzęt i warunki
   Format docelowy: WAV, 16 kHz, mono. Nagrania z telefonu (.m4a) konwertujesz poleceniem:
   bash
   ffmpeg -i nagranie.m4a -ar 16000 -ac 1 q09_B_kacper.wav

2. Ile nagrywać
   ┌───────────────────┬───────────┬────────┐
   │ Pytanie │ Warianty │ Liczba │
   ├───────────────────┼───────────┼────────┤
   │ q01 (o sobie) │ A, B │ 2 │
   │ q02 (projekt) │ A, B │ 2 │
   │ q07 (Scrum) │ A, B, C │ 3 │
   │ q09 (REST API) │ A, B, C │ 3 │
   │ q11 (Docker) │ A, B, C │ 3 │
   │ q15 (PoC) │ A, B, C │ 3 │
   ├───────────────────┼───────────┼────────┤
   │ Razem │ │ 16 │
   └───────────────────┴───────────┴────────┘

To wystarczy, żeby zbudować i przetestować cały przepływ.

Etap 2 (pełny zbiór, ok. 54 nagrania na osobę)
┌──────────────────────────────────┬──────────┬──────────┐
│ Pytania │ Warianty │ Liczba │
├──────────────────────────────────┼──────────┼──────────┤
│ q01–q06 (ogólne i behawioralne) │ A, B │ 12 │
│ q07–q20 (procesowe i techniczne) │ A, B, C │ 42 │
│ opcjonalnie: wybrane techniczne │ D │ dowolnie │
├──────────────────────────────────┼──────────┼──────────┤
│ Razem │ │ 54+ │
└──────────────────────────────────┴──────────┴──────────┘

Liczba osób: im więcej różnych głosów, tym lepiej. Cel minimum to 2–3 osoby, przynajmniej dla Etapu 1.

Długość jednej odpowiedzi

45–120 sekund, najlepiej ok. 60–90 s. Krótsze nagrania dają niewiarygodne statystyki (tempo, TTR).

3. Warianty
   ┌─────────┬──────────┬────────────┬───────────────────────────────────┐
   │ Wariant │ Styl │ Płynność │ Treść │
   ├─────────┼──────────┼────────────┼───────────────────────────────────┤
   │ A │ formalny │ płynny │ poprawna │
   │ B │ potoczny │ przerywany │ poprawna │
   │ C │ formalny │ płynny │ częściowa / błędna / nie na temat │
   │ D │ potoczny │ przerywany │ częściowa / błędna │
   └─────────┴──────────┴────────────┴───────────────────────────────────┘

Dla q01–q06 nie ma poprawnej odpowiedzi, więc nagrywasz tylko A i B (różnią się formą).

A: wzorcowy kandydat
Pełne zdania, słownictwo branżowe, uporządkowana odpowiedź.
Naturalnie użyj kilku terminów z kolumny expected_terms danego pytania.
Pojedyncze naturalne zawahanie jest OK, nie musisz brzmieć jak lektor.
B: wie, ale słabo mówi
Treść poprawna, ale język luźny: „spoko”, „ogarnąć”, „no i generalnie”, „jakby”, „ten tego”, „w sumie”.
Przerywniki („yyy”, „eee”, „no”), dłuższe pauzy, powtórzenia („to jest… to jest takie”), urwane i zaczynane od nowa zdania.
Terminy wymawiaj po polsku, tak jak w mowie potocznej (patrz sekcja 4).
C: ładnie leje wodę
Brzmi profesjonalnie i płynnie, ale treść jest słaba. Wybierz jedno:
ogólnikowo: dużo słów, mało konkretów („to bardzo ważna technologia, która usprawnia procesy”),
z błędem: np. pomylenie uwierzytelniania z autoryzacją, Dockera z maszyną wirtualną,
nie na temat: zaczynasz od pytania i odpływasz w inny temat.
W kolumnie notes zapisz, jaki błąd lub odpływ był zamierzony.
D: najgorszy przypadek (opcjonalnie)

Połączenie B i C: potocznie, z zawahaniami i słabo merytorycznie.

Ogólne zasady
Nie czytaj z kartki. Czytany tekst brzmi nienaturalnie i ma inne zawahania niż mowa spontaniczna. Możesz mieć przed sobą 3–4 słowa kluczowe.
Nie przesadzaj z graniem. Lekko przerysowane jest OK, karykatura nie.
Jeśli nagranie wyszło inaczej, niż planowałeś, nie powtarzaj go. Zostaw je i opisz zgodnie z rzeczywistością w kolumnach label\_\*. Takie przypadki też są cenne. 4. Terminy do wplatania (szczególnie w wariantach B i D)

Celowo wymawiaj je „po polsku”, żeby sprawdzić korektę transkrypcji:

┌─────────┬────────────────────────────────────────┬────────────────────────────────────┐
│ Pytanie │ Wymowa potoczna │ Forma poprawna │
├─────────┼────────────────────────────────────────┼────────────────────────────────────┤
│ q07 │ dżira, skram, dejli, bakklog │ Jira, Scrum, daily, backlog │
│ q08 │ pul rekłest, kołd riwju, merdżować │ pull request, code review, merge │
│ q09 │ rest-api, dżej-son, endpojnt │ REST API, JSON, endpoint │
│ q11 │ doker, dokerfajl, doker kompołz │ Docker, Dockerfile, docker-compose │
│ q12 │ ci-aj-si-di, gitchab ekszyns, dżenkins │ CI/CD, GitHub Actions, Jenkins │
│ q13 │ gita, brancz, rebejs │ Git, branch, rebase │
│ q15 │ pe-o-ka, POK, pruf of koncept │ PoC, proof of concept │
│ q18 │ dżej-dabl-ju-ti, o-aut │ JWT, OAuth │
│ q19 │ kubernetis, ku-ejts, jamel │ Kubernetes, K8s, YAML │
└─────────┴────────────────────────────────────────┴────────────────────────────────────┘ 5. Nazwy plików
<question*id>*<wariant>\_<osoba>.wav

Przykłady: q09_B_kacper.wav, q15_C_ania.wav. Jeśli nagrywasz ten sam wariant drugi raz: q09_B_kacper_2.wav.

Pliki trafiają do data/audio/.

6. Wypełnianie recordings.csv

Od razu po nagraniu uzupełnij: file, question_id, speaker, variant, intended_style, intended_fluency, intended_content, duration_s, recorded_at, notes.

Po odsłuchaniu (najlepiej przez inną osobę albo przynajmniej następnego dnia) uzupełnij label_style, label_fluency, label_content. To etykiety prawdy do liczenia precision/recall/F1. Opisują to, co faktycznie słychać, a nie to, co planowałeś.

Dozwolone wartości:

┌───────────┬───────────────────────────────────────────────────────────────┐
│ Kolumna │ Wartości │
├───────────┼───────────────────────────────────────────────────────────────┤
│ _\_style │ formalny, potoczny │
│ _\_fluency │ płynny, przerywany │
│ \*\_content │ poprawna, częściowa, błędna, nie_na_temat (puste dla q01–q06) │
└───────────┴───────────────────────────────────────────────────────────────┘

Przykład:

csv
q09_B_kacper.wav,q09,kacper,B,potoczny,przerywany,poprawna,potoczny,przerywany,poprawna,74,2026-11-01,"mówiłem ""dżej-son"" zamiast JSON"

7. Transkrypcje wzorcowe

Dla każdego nagrania utwórz plik tekstowy o tej samej nazwie w data/transcripts/, np. q09_B_kacper.txt. To z nimi porównujemy wynik Whispera (CER).

Zasady zapisu:

Zapisuj dokładnie to, co powiedziano, łącznie z powtórzeniami, urwanymi słowami i kolokwializmami.
Przerywniki zawsze w ten sam sposób: yyy, eee, mmm. „No”, „jakby”, „w sumie” zapisuj normalnie jako słowa.
Terminy branżowe zapisuj w formie poprawnej (Jira, PoC, JSON), nawet jeśli wymówione były potocznie, bo to jest cel korekty.
Liczby zapisuj słownie, tak jak zostały wypowiedziane.
Bez znaczników pauz. Pauzy liczymy z timestampów.
