import pandas as pd
from process import build_hotwords


def test_strips_whitespace_and_skips_duplicates():

    terms = pd.DataFrame({"term": [" Jira ", "Jira", "Docker", ""]})

    results = build_hotwords(terms)

    assert results == ["Jira", "Docker"]


def test_keeps_csv_order():

    terms = pd.DataFrame({"term": ["Jira", "Redis", "Docker"]})

    results = build_hotwords(terms)
     
    assert results == ["Jira", "Redis", "Docker"]
    