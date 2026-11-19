import pandas as pd 
from process import build_hotwords
import pytest

def test_excludes_hidden_terms():

    terms = pd.DataFrame({"term": ["Jira", "Redis", "Docker", "Pycharm"]})

    hidden_terms = ["Redis"]

    results = build_hotwords(terms, hidden_terms)

    assert results == ["Jira", "Docker", "Pycharm"]

def test_strips_whitespace_and_skips_duplicates():

    terms = pd.DataFrame({"term": [" Jira ", "Jira", "Docker", ""]})

    hidden_terms = []

    results = build_hotwords(terms, hidden_terms)

    assert results == ["Jira", "Docker"]

def test_raises_for_hidden_term_not_in_dictionary():

    terms = pd.DataFrame({"term": ["Jira", "Docker"]})

    hidden_terms = ["Terraform"]

    with pytest.raises(ValueError, match="Terraform"):
        build_hotwords(terms, hidden_terms)

def test_raises_for_hidden_term_with_different_case():

    terms = pd.DataFrame({"term": ["Jira", "Redis", "Docker"]})

    hidden_terms = ["redis"]

    with pytest.raises(ValueError, match="redis"):
        build_hotwords(terms, hidden_terms)

def test_returns_all_terms_when_no_hidden():

    terms = pd.DataFrame({"term": ["Jira", "Redis", "Docker"]})

    hidden_terms = []

    results = build_hotwords(terms, hidden_terms)
     
    assert results == ["Jira", "Redis", "Docker"]