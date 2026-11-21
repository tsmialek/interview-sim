import pytest

from process import count_hotword_tokens, ensure_hotwords_fit, get_dropped_hotwords


class FakeTokenizer:
    """1 word = 1 token, so limits can be counted by hand.

    With max_length=8 the limit is 8 // 2 - 1 = 3 tokens.
    """

    def encode(self, text):
        return text.split()


def test_counts_tokens_of_joined_hotwords():
    hotwords = ["Jira", "Docker", "Microsoft Teams"]

    results = count_hotword_tokens(hotwords, FakeTokenizer())

    assert results == 4


def test_returns_empty_when_list_exactly_fits():
    hotwords = ["A", "B", "C"]

    results = get_dropped_hotwords(hotwords, FakeTokenizer(), max_length=8)

    assert results == []


def test_drops_terms_over_limit():
    hotwords = ["A", "B", "C", "D", "E"]

    results = get_dropped_hotwords(hotwords, FakeTokenizer(), max_length=8)

    assert results == ["D", "E"]


def test_reports_partially_cut_term_as_dropped():
    hotwords = ["A", "B", "Microsoft Teams", "C"]

    results = get_dropped_hotwords(hotwords, FakeTokenizer(), max_length=8)

    assert results == ["Microsoft Teams", "C"]


def test_passes_when_no_hotwords():
    results = ensure_hotwords_fit(None, FakeTokenizer(), max_length=8)

    assert results is None


def test_passes_when_hotwords_fit():
    hotwords = ["Jira", "Docker", "Redis"]

    results = ensure_hotwords_fit(hotwords, FakeTokenizer(), max_length=8)

    assert results is None


def test_raises_when_hotwords_are_cut():
    hotwords = ["Jira", "Docker", "Redis", "Kubernetes"]

    with pytest.raises(ValueError, match="Kubernetes"):
        ensure_hotwords_fit(hotwords, FakeTokenizer(), max_length=8)


def test_raises_for_string_hotwords_over_limit():
    with pytest.raises(ValueError):
        ensure_hotwords_fit(
            "Jira Docker Redis Kubernetes", FakeTokenizer(), max_length=8
        )


def test_passes_for_string_hotwords_that_fit():
    results = ensure_hotwords_fit("Jira Docker Redis", FakeTokenizer(), max_length=8)

    assert results is None
