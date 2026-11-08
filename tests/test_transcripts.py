from transcripts import DATA_DIR, load_term_names, normalize, parse_transcript, plain_text


def test_term_position():
    words, terms = parse_transcript("zrobiliśmy {deploy|deploy} w piątek")
    assert words == ["zrobiliśmy", "deploy", "w", "piątek"]
    assert terms == [{"term": "deploy", "form": "deploy", "start": 1, "end": 2}]


def test_multiword_term():
    words, terms = parse_transcript("chowamy za {feature flagą|feature flag}.")
    assert terms == [{"term": "feature flag", "form": "feature flagą", "start": 2, "end": 4}]
    assert words[2:4] == ["feature", "flagą"]


def test_apostrophe_and_hyphen_in_form():
    _, terms = parse_transcript("projektu {e-commerce'owego|e-commerce}")
    assert terms[0]["form"] == "ecommerceowego"


def test_separators_inside_word_are_joined():
    assert normalize("pg-vector, CI/CD i Next.js") == ["pgvector", "cicd", "i", "nextjs"]


def test_cut_off_word_keeps_its_hyphen():
    assert normalize("dużego po- Pytona") == ["dużego", "po-", "pytona"]


def test_punctuation_after_term():
    words, _ = parse_transcript("do {Grafany|Grafana}, i od razu")
    assert words == ["do", "grafany", "i", "od", "razu"]


def test_fillers_do_not_shift_positions():
    words, terms = parse_transcript("yyy {deploy|deploy} eee")
    assert words == ["deploy"]
    assert (terms[0]["start"], terms[0]["end"]) == (0, 1)


def test_version_numbers_spelled_out():
    assert normalize("wersja 1.4.1, potem 1.3") == [
        "wersja", "jeden", "kropka", "cztery", "kropka", "jeden", "potem", "jeden", "kropka", "trzy",
    ]


def test_term_names_by_category():
    names = load_term_names({"tool", "technology"})
    assert "Vocare" in names and "Hangfire" in names
    assert "branch" not in names
    assert len(names) < len(load_term_names())


def test_parser_words_match_wer_normalization():
    # term positions only line up with the WER alignment if both see the same words
    for path in sorted((DATA_DIR / "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        words, terms = parse_transcript(text)
        assert words == normalize(plain_text(text)), path.name
        for t in terms:
            assert " ".join(words[t["start"]:t["end"]]) == t["form"], path.name
