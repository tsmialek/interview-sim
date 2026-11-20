from validate_data import find_missing_hidden_terms


def test_returns_empty_when_all_hidden_terms_are_in_dictionary():
    terms = ["Jenkins", "Redis", "Docker"]
    hidden_terms = ["Jenkins", "Redis"]

    results = find_missing_hidden_terms(terms, hidden_terms)

    assert results == []


def test_finds_hidden_term_with_typo():
    terms = ["Jenkins", "Redis"]
    hidden_terms = ["Jenkis"]

    results = find_missing_hidden_terms(terms, hidden_terms)

    assert results == ["Jenkis"]


def test_ignores_case():
    terms = ["Jenkins", "Redis"]
    hidden_terms = ["redis"]

    results = find_missing_hidden_terms(terms, hidden_terms)

    assert results == []


def test_returns_all_missing_terms_in_file_order():
    terms = ["Redis"]
    hidden_terms = ["Kanban", "Redis", "Terraform"]

    results = find_missing_hidden_terms(terms, hidden_terms)

    assert results == ["Kanban", "Terraform"]


def test_returns_empty_when_no_hidden_terms():
    terms = ["Jenkins"]
    hidden_terms = []

    results = find_missing_hidden_terms(terms, hidden_terms)

    assert results == []
