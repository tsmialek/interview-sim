from process import count_hotword_tokens, get_dropped_hotwords

class FakeTokenizer:
    def encode(self, text):
        return text.split()

def test_counts_tokens_of_joined_hotwords():

    results = count_hotword_tokens(["Jira", "Docker", "Microsoft Teams"], FakeTokenizer())

    assert results == 4

def test_returns_empty_when_list_exactly_fits():

    results = get_dropped_hotwords(["A", "B", "C"], FakeTokenizer(), max_length=8)

    assert results == []

def test_drops_terms_over_limit():
    results = get_dropped_hotwords(["A", "B", "C", "D", "E"], FakeTokenizer(), max_length=8)

    assert results == ["D", "E"]

def test_reports_partially_cut_term_as_dropped():
    results = get_dropped_hotwords(["A", "B", "Microsoft Teams", "C"], FakeTokenizer(), max_length=8)

    assert results == ["Microsoft Teams", "C"]