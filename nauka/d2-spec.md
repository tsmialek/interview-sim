# Specyfikacja datasetu D2 (trening HerBERT)

Task: [E2] Specyfikacja datasetu D2 (trening HerBERT).

HerBERT uczy się wskazywać, **gdzie** w zdaniu jest termin IT, także spoza słownika. Jak termin zapisać, ustala słownik albo SLM.

## 1. Format i etykiety

### 1.1. Plik `data/ner/train.txt`

- Jedno zdanie w wierszu, kodowanie UTF-8.
- Termin w klamrach `{forma|termin}`, tak samo jak we wzorcach w `data/transcripts/`:
  - `forma` to tekst, który stoi w zdaniu: odmieniony, a po zepsuciu (sekcja 3) zapisany jak u Whispera (`dżirze`);
  - `termin` to forma podstawowa z `data/terms_it.csv` albo nowy termin spoza słownika.
- Zdanie-negatyw nie ma klamer.

```
wrzuciłem {pul rekłest|pull request} na {githaba|GitHub}
logi zbieramy w {Kibanie|Kibana} i potem patrzymy na wykresy
w zoo widziałem pytona
```

Model dostaje tylko tekst z lewej strony klamry. Prawa strona służy do kontroli danych: zakazu terminów z `hidden_terms.txt` i statystyk.

### 1.2. Konwersja na etykiety BIO

Zdanie dzielimy na słowa po spacjach. Każde słowo dostaje jedną etykietę:

| etykieta | znaczenie |
|---|---|
| `B-TERM` | pierwsze słowo terminu |
| `I-TERM` | kolejne słowo tego samego terminu |
| `O` | słowo poza terminem |

```
wrzuciłem  O
pul        B-TERM
rekłest    I-TERM
na         O
githaba    B-TERM
```

BIO nie jest zapisywane w pliku. Powstaje z klamer w kodzie treningu, więc `train.txt` zostaje czytelny dla człowieka.

### 1.3. Dopasowanie do tokenów pod-słowowych HerBERT

Tokenizer HerBERT (`allegro/herbert-base-cased`) tnie słowa na kawałki. Poniżej prawdziwy wynik tokenizera:

```
wrzuciłem  pul    rekłest      na   githaba
wrzu|ciłem pu|l   rek|łe|st    na   gi|tha|ba
```

**Reguła: etykietę słowa dostaje tylko jego pierwszy kawałek.** Pozostałe kawałki dostają `-100` i nie liczą się do błędu przy treningu. Tokeny specjalne `<s>` i `</s>` też dostają `-100`.

| kawałek | wrzu | ciłem | pu | l | rek | łe | st | na | gi | tha | ba |
|---|---|---|---|---|---|---|---|---|---|---|---|
| etykieta | O | -100 | B-TERM | -100 | I-TERM | -100 | -100 | O | B-TERM | -100 | -100 |

To, z którego słowa pochodzi kawałek, podaje `word_ids()` tokenizera HuggingFace (dla tego zdania: `[None, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, None]`).

**Uzasadnienie.** Polskie odmiany i zepsute aliasy tną się na wiele kawałków: `Kubernetesie` na 5, `dżirze` na 3, a `na` na 1. Gdyby etykietę dostawał każdy kawałek, długie słowa liczyłyby się do błędu wielokrotnie. Przy tej regule każde słowo waży tyle samo. Ocena i tak odbywa się na poziomie słów.

Uwaga techniczna: tokenizer HerBERT wymaga pakietu `sacremoses`, którego nie ma w `environment.yml`. Trzeba go dodać przy tasku treningu.

## 2. Prompty generujące

Dwa osobne prompty: jeden dla zdań z terminami (pozytywy), drugi dla negatywów. Dzięki temu proporcję negatywów (sekcja 4) ustawiamy sami, liczbą wywołań.

Wspólne zasady:

- **Styl mówiony bez przerywników.** HerBERT czyta tekst z Whispera, a Whisper zwykle wycina `yyy`/`eee`. Zostawia natomiast powtórzenia, urwane myśli i potoczne „no”, „jakby”, „wiesz”. Takie zdania generujemy.
- **Ukrytych terminów nie ma w prompcie.** Nie wymieniamy ich nawet jako zakazu, bo sama nazwa w prompcie zwiększa szansę, że model jej użyje. LLM i tak może użyć ich sam z siebie (`deploy`, `merge` są bardzo częste), dlatego po generacji filtrujemy wynik (sekcja 4).
- **Różnorodność.** Jedno wywołanie to jedno pytanie rekrutacyjne i 20 zdań. Pytania rotujemy: 15 kart z Załącznika A oraz kolejne pytania techniczne. Temperatura ok. 0.9.
- **Pola `{…}`** w promptach wypełnia skrypt generujący.
- **Model:** próbka z sekcji 7 powstała w Claude Opus 5.5 (`claude-opus-5-5`). Model do pełnego zbioru wybieramy po przeglądzie próbki.
- **Skróty po lewej stronie klamry są dozwolone** (`{PR|pull request}`). Tak się mówi, a HerBERT ma znaleźć termin w takiej formie, w jakiej padł.

### 2.1. Prompt: zdania z terminami

Pole `{TERMINY_ZE_SLOWNIKA}` to kolumna `term` z `data/terms_it.csv` **bez** terminów z `data/hidden_terms.txt`, rozdzielona przecinkami.

```
Generujesz dane treningowe do modelu, który wykrywa terminy IT w polskich transkrypcjach rozmów rekrutacyjnych.

Napisz {LICZBA} zdań, które kandydat mógłby powiedzieć na głos, odpowiadając na pytanie:
„{PYTANIE}”

Styl:
- mowa, nie tekst pisany: luźna składnia, powtórzenia („to, to było”), urwane myśli, wtrącenia „no”, „jakby”, „wiesz”, „w sumie”,
- bez „yyy”, „eee”, „mmm”,
- zdania różnej długości (od 6 do 30 słów), różni mówcy, różne poziomy doświadczenia.

Terminy:
- każde zdanie ma od 1 do 3 terminów IT,
- część terminów weź z tej listy: {TERMINY_ZE_SLOWNIKA},
- co najmniej połowa terminów ma być SPOZA tej listy: inne narzędzia, biblioteki, języki, usługi chmurowe, pojęcia (np. GraphQL, Kibana, Vue, Celery, gRPC, load balancer, sharding),
- terminy odmieniaj tak, jak mówi się po polsku: „na Kubernetesie”, „w Kibanie”, „PR-a”, „z GraphQL-em”, „dockerowy”,
- terminy wielowyrazowe też są w porządku („load balancer”, „code review”).

Format (ściśle):
- jedno zdanie w wierszu, bez numeracji, bez cudzysłowów, bez komentarzy,
- każdy termin w klamrach {forma|termin}:
  - forma: termin zapisany poprawnie, w formie z tego zdania (odmieniony),
  - termin: forma podstawowa, kanoniczna pisownia (np. „Kibana”, „GraphQL”, „load balancer”),
- nic poza terminami IT nie trafia w klamry.

Przykłady:
no i całe logi zbieraliśmy w {Kibanie|Kibana}, bo jakby inaczej to szukanie błędu trwało wieki
w sumie {API|API} wystawialiśmy w {GraphQL-u|GraphQL}, a nie po staremu
na początku to, to wszystko stało na jednym serwerze, dopiero potem doszedł {load balancer|load balancer}
wiesz, każdy {PR|pull request} musiał przejść {code review|code review} u dwóch osób
```

### 2.2. Prompt: zdania-negatywy

Pole `{SLOWA_PULAPKI}` to lista z Załącznika B dokumentu `dataset-krok-po-kroku.md`: doker, brancz, sequel, pyton, komitet, reakcja, POK, szlak, na jawie, kafle, grafika, sprint (bieg), ażurowy, rest (reszta).

```
Generujesz dane treningowe do modelu, który wykrywa terminy IT w polskich transkrypcjach. Teraz potrzebuję zdań BEZ żadnych terminów IT.

Napisz {LICZBA} zdań, które kandydat mógłby powiedzieć na głos na rozmowie rekrutacyjnej, odpowiadając na pytanie nietechniczne:
„{PYTANIE}”

Styl:
- mowa, nie tekst pisany: luźna składnia, powtórzenia, urwane myśli, wtrącenia „no”, „jakby”, „wiesz”,
- bez „yyy”, „eee”, „mmm”,
- zdania od 6 do 30 słów.

Treść:
- w każdym zdaniu użyj co najmniej jednego słowa z listy, w ZWYKŁYM, nie-informatycznym znaczeniu: {SLOWA_PULAPKI},
- odmieniaj te słowa: „dokerem”, „pytona”, „kaflami”, „na branczu”, „w komitecie”,
- możesz użyć też innych zwykłych polskich słów, które brzmią podobnie do terminów IT,
- żadnych terminów IT, żadnych klamer.

Format: jedno zdanie w wierszu, bez numeracji, bez cudzysłowów, bez komentarzy.

Przykłady:
mój dziadek przez trzydzieści lat był dokerem w Gdyni, no i stąd ta miłość do statków
w zoo to najbardziej zapamiętałem pytona, taki był ogromny, że aż się bałem
w weekend kładliśmy z bratem nowe kafle w łazience i to, to było cięższe niż myślałem
```

Pytania nietechniczne: N1–N6 z Załącznika B oraz kolejne w tym stylu (hobby, motywacja, stres, wakacje).

## 3. Psucie terminów na aliasy (plan skryptu)

LLM zapisuje terminy poprawnie: `{Jirze|Jira}`. Whisper często zapisuje je fonetycznie: `dżirze`. Skrypt zamienia lewą stronę części klamer na taki zapis, żeby HerBERT uczył się na tekście podobnym do wyjścia Whispera.

**Kolejność:** generacja → psucie → filtr ukrytych terminów (sekcja 4) → zapis `train.txt`. BIO powstaje dopiero z gotowego pliku (sekcja 1.2), więc to, że alias ma inną liczbę słów niż forma (`PR` → `pi ar`), nie jest problemem.

### 3.1. Zasady

1. **Psujemy ok. 50% klamer**, losowo, ze stałym ziarnem (`seed`), żeby wynik dało się odtworzyć. Druga połowa zostaje poprawna, bo Whisper z hotwords część terminów zapisuje dobrze. HerBERT ma rozpoznać obie wersje.
2. **Zmienia się tylko lewa strona.** Prawa (`termin`) zostaje bez zmian.
3. **Termin ze słownika:** alias z kolumny `aliases` w `data/terms_it.csv`. Wybieramy alias z **najdłuższą wspólną końcówką** z formą w zdaniu, żeby zachować przypadek:

   | forma w zdaniu | aliasy `Jira` | wybrany | wspólna końcówka |
   |---|---|---|---|
   | `Jirze` | dżira, dżiry, dżirze, dżirą, dżirę, jiry, jirze | `dżirze` lub `jirze` | `irze` |
   | `Jirą` | j.w. | `dżirą` | `irą` |

   Remis rozstrzyga losowanie. Jeśli najlepszy alias ma mniej niż 2 wspólne znaki końcowe, klamra zostaje niezepsuta. Przykład z próbki: dla `{API|API}` jedynym kandydatem był `apiki` (wspólne tylko `i`), czyli zmiana liczby i przypadku.
4. **Termin spoza słownika:** brak aliasów, więc stosujemy reguły spolszczania (3.2).

### 3.2. Reguły spolszczania (terminy spoza słownika)

Reguły stosujemy po kolei do lewej strony klamry:

| # | reguła | przykład |
|---|---|---|
| 1 | małe litery | `Kibanie` → `kibanie` |
| 2 | usuń łącznik przed końcówką | `GraphQL-em` → `graphqlem` |
| 3 | `ql` → `kuel` | `graphqlem` → `graphkuelem` |
| 4 | `ph` → `f` | `graphkuelem` → `grafkuelem` |
| 5 | `qu` przed samogłoską → `kw` | `queue` → `kweue`; `SonarQube` bez zmian (wcześniej dawało `sonarkwbe`) |
| 6 | `sh` → `sz` | `sharding` → `szarding` |
| 7 | `th` → `t` | `thread` → `tread` |
| 8 | `c` przed `e`, `i`, `y` → `s` | `balancer` → `balanser`, `Celery` → `selery` |
| 9 | `x` → `ks` | `nginx` → `nginks` |
| 10 | `y` na końcu słowa → `i` | `selery` → `seleri` |
| 11 | `j` na początku słowa → `dż` | `Jest` → `dżest` |

Reguły są celowo proste: to przybliżenie błędów Whispera, a nie wierny model wymowy. Jeśli po regułach forma się nie zmieniła (np. `kafce` już spolszczone), klamrę liczymy jako poprawną.

Na próbce (sekcja 7) sprawdzamy, czy aliasy brzmią sensownie. Reguły, które dają dziwne wyniki, usuwamy lub poprawiamy.

### 3.3. Przykład

```
przed:  no i logi zbieraliśmy w {Kibanie|Kibana}, a taski mieliśmy w {Jirze|Jira}
po:     no i logi zbieraliśmy w {kibanie|Kibana}, a taski mieliśmy w {dżirze|Jira}
```

## 4. Negatywy i ukryte terminy

### 4.1. Proporcja negatywów: 25%

Negatyw to zdanie ze słowem-pułapką w zwykłym znaczeniu, bez klamer (prompt 2.2). Bez negatywów model uczy się reguły „brzmi jak alias = termin”, czyli robi to samo co słownik. Negatywy uczą, że o tym, czy słowo jest terminem, decyduje kontekst.

25% to środek widełek 20–30% z taska. Proporcję liczymy **po** filtrze z 4.2. Jeśli filtr usunie część pozytywów, dogenerowujemy pozytywy, a nie usuwamy negatywów.

### 4.2. Filtr ukrytych terminów

Terminy z `data/hidden_terms.txt` (Jenkins, Terraform, Redis, Azure, Scrum, Kanban, deploy, merge, REST, DevOps) nie mogą wystąpić w treningu. W ocenie E1 mierzą recall terminów spoza słownika. Gdyby HerBERT je widział, wynik byłby zawyżony.

**Zdanie z ukrytym terminem usuwamy w całości.** Nie wystarczy zdjąć z niego klamer, bo wtedy termin dostałby etykietę `O` i uczylibyśmy model, że to **nie** jest termin.

| zbiór | zdanie odrzucamy, gdy | dopasowanie |
|---|---|---|
| pozytywy | ukryty termin jest po prawej stronie klamry | dokładnie, bez wielkości liter |
| pozytywy | ukryty termin **lub jego alias** z `terms_it.csv` występuje w tekście poza klamrami (`dewopsa`, `ażurze`, `deployu`) | całe słowa, bez wielkości liter |
| negatywy | ukryty termin występuje w tekście | całe słowa, **z** wielkością liter, **bez** aliasów |

Dlaczego negatywy mają łagodniejszy filtr: słowa-pułapki `ażurowy` i `rest` (reszta) celowo przypominają `Azure` i `REST`. Sprawdzanie aliasów (`ażur`, `rest`) odrzuciłoby dokładnie te zdania, których potrzebujemy.

Znany wyjątek: `merge request` jest w słowniku i nie jest ukryty, choć zawiera słowo `merge`. Takie klamry zostają, bo prawa strona to `merge request`, a nie `merge`. Ryzyko przecieku jest małe i akceptujemy je.

## 5. Docelowa wielkość zbioru: 2000 zdań

1500 pozytywów i 500 negatywów (25%, sekcja 4.1). Mieści się w widełkach 1000–3000 z `dataset-krok-po-kroku.md`.

Uzasadnienie:

- **Terminy ze słownika:** ok. 85 terminów (bez ukrytych) dostaje po 10–15 wystąpień w różnych odmianach i kontekstach.
- **Terminy spoza słownika:** co najmniej połowa wystąpień (prompt 2.1), czyli kilkaset **różnych** terminów. Dzięki temu model uczy się rozpoznawać termin po kontekście i kształcie słowa, a nie zapamiętuje listy.
- **Skala:** fine-tuning modeli typu BERT do NER zwykle daje dobre wyniki już przy kilku tysiącach zdań.
- **Górna granica:** więcej zdań z LLM to więcej tego samego czystego stylu (sekcja 6). Kolejne tysiące zdań poprawią wynik na D2, ale niekoniecznie na transkrypcjach.

Jeśli wynik na D1 będzie słaby, najpierw sprawdzamy różnorodność terminów i jakość aliasów, a dopiero potem zwiększamy liczbę zdań.

## 6. Ryzyko: D2 jest czystsze niż tekst Whispera

Zdania z LLM mają poprawną interpunkcję, pełne zdania i przewidywalne błędy (tylko te z sekcji 3). Tekst z Whispera ma inne błędy: sklejone i rozbite słowa, pominięte fragmenty, błędy niezwiązane z terminami. Wynik HerBERT na D2 będzie więc **wyższy** niż na prawdziwych transkrypcjach.

Dlatego:

- **Test końcowy zawsze na transkrypcjach D1, nigdy na D2.**
- D2 dzielimy tylko na trening i walidację (np. 90/10), żeby wiedzieć, kiedy przerwać trening. Wynik walidacji na D2 nie trafia do raportu jako wynik modelu.

## 7. Próbka do przeglądu

Plik: `data/ner/sample.txt`, 50 zdań: 38 pozytywów (wiersze 1–38) i 12 negatywów (wiersze 39–50). Zdania wygenerował Claude Opus 5.5 według promptów z sekcji 2, a potem zostały zepsute według reguł z sekcji 3 (`seed=42`).

| miara | wartość |
|---|---|
| klamry razem | 66 |
| terminy ze słownika / spoza słownika | 28 / 38 (58% spoza) |
| klamry zepsute | 26 z 66 (39%) |
| ukryte terminy (filtr z 4.2) | 0 |
| słowa-pułapki w negatywach | wszystkie 14 z Załącznika B |

Zepsutych klamer jest mniej niż 50%, bo część wylosowanych klamer nie dała się zmienić: terminy spoza słownika już pisane małymi literami, do których nie pasuje żadna reguła (`pipeline`, `deadlock`), oraz terminy ze słownika bez aliasu o pasującej końcówce (`API`). Dla pełnego zbioru można podnieść prawdopodobieństwo psucia do ok. 0.6.

Na co patrzeć przy przeglądzie:

1. Czy zdania brzmią jak mowa na rozmowie, a nie jak tekst pisany.
2. Czy aliasy wyglądają jak błędy Whispera (`grafkuelu`, `seleri`, `szarding`, `dockerze`).
3. Czy nie ma terminów IT bez klamer (np. „serwer”, „baza” zostawiliśmy jako zwykłe słowa).
4. Czy negatywy są jednoznacznie nie-IT.
