import pandas as pd

from log_runs import DictionaryConfig
from process import apply_replacements, find_dictionary_replacements

DICTIONARY = pd.DataFrame(
    {
        "term": ["Jira", ".NET", "Docker", "pull request", "request", "Confluence"],
        "aliases": [
            ["dżira", "dżirze"],
            ["dot net", "dotnet"],
            ["doker", "dokera"],
            ["pul rikłest", "pull requesta"],
            ["request"],
            ["konfluensie"],
        ],
    }
)
CONFIG = DictionaryConfig(
    terms_sha="test",
    hidden_terms_sha="test",
    rapidfuzz_threshold=85,
)


def find(text):
    return find_dictionary_replacements(text, DICTIONARY, CONFIG)


def test_replaces_single_word_alias():
    text = "mamy taski w dżirze"

    results = find(text)

    assert len(results) == 1
    assert results[0].r_from == "dżirze"
    assert results[0].r_to == "Jira"
    assert results[0].base_term == "Jira"
    assert results[0].position == text.index("dżirze")
    assert results[0].source == "dictionary"


def test_replaces_multi_word_alias():
    text = "piszemy w dot net"

    results = find(text)

    assert len(results) == 1
    assert results[0].r_from == "dot net"
    assert results[0].r_to == ".NET"
    assert results[0].position == text.index("dot net")


def test_does_not_replace_inside_another_word():
    results = find("projekt dotnetowy")

    assert results == []


def test_keeps_case_of_r_from():
    results = find("Dżirze ufamy")

    assert results[0].r_from == "Dżirze"
    assert results[0].position == 0


def test_replaces_fuzzy_match_above_threshold():
    results = find("w konfluensje")

    assert results[0].r_from == "konfluensje"
    assert results[0].r_to == "Confluence"


def test_ignores_fuzzy_match_below_threshold():
    results = find("w dżiże")

    assert results == []


def test_leaves_inflected_term_unchanged():
    results = find("kontener Dockera")

    assert results == []


def test_correct_form_protects_its_words():
    results = find("zrobiłem pull requesta")

    assert results == []


def test_does_not_join_words_across_punctuation():
    results = find("dot. Net")

    assert results == []


def test_returns_replacements_sorted_by_position():
    text = "dżirze i dot net"

    results = find(text)

    assert [r.position for r in results] == [0, text.index("dot net")]


def test_positions_fit_apply_replacements():
    text = "obraz dokera w dżirze"

    results = apply_replacements(text, find(text))

    assert results == "obraz Docker w Jira"
