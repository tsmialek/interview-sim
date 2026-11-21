import pandas as pd
import pytest

from log_runs import RunConfig
from run_variant import HOTWORDS_SOURCE, run_variant_second


class FakeProject:
    """Stands in for ProjectRepository; hides terms case-insensitively like get_dictionary."""

    def __init__(self, terms, hidden_terms):
        self.terms = terms
        self.hidden_terms = hidden_terms

    def get_hidden_terms(self):
        return self.hidden_terms

    def get_dictionary(self, exclude_hidden=True):
        hidden = {term.lower() for term in self.hidden_terms} if exclude_hidden else set()
        return pd.DataFrame({"term": [term for term in self.terms if term.lower() not in hidden]})


@pytest.fixture
def run_first(mocker):
    """Replaces run_variant_first, so no model is loaded and nothing is saved."""
    return mocker.patch("run_variant.run_variant_first", return_value="run-1")


@pytest.fixture
def variant_one_config(default_whisper_config):
    return RunConfig(variant=1, whisper=default_whisper_config)


def test_runs_variant_one_with_hotwords_without_hidden_terms(run_first, variant_one_config):
    project = FakeProject(["Jira", "Redis", "Docker"], ["redis"])
    repo = object()

    results = run_variant_second(variant_one_config, "dev", repo, project)

    assert results == "run-1"
    config, split, passed_run_repo, passed_project_repo = run_first.call_args.args
    assert split == "dev"
    assert passed_run_repo is repo
    assert passed_project_repo is project
    assert config.variant == 2
    assert config.whisper.hotwords == ["Jira", "Docker"]
    assert config.whisper.hotwords_source == HOTWORDS_SOURCE


def test_keeps_other_whisper_settings(run_first, variant_one_config):
    project = FakeProject(["Jira"], [])

    run_variant_second(variant_one_config, "dev", object(), project)

    config = run_first.call_args.args[0]
    only_hotwords = {"hotwords", "hotwords_source"}
    assert config.whisper.model_dump(exclude=only_hotwords) == variant_one_config.whisper.model_dump(
        exclude=only_hotwords
    )


def test_raises_for_hidden_term_not_in_dictionary(run_first, variant_one_config):
    project = FakeProject(["Jenkins", "Redis"], ["Jenkis"])

    with pytest.raises(ValueError, match="Jenkis"):
        run_variant_second(variant_one_config, "dev", object(), project)

    run_first.assert_not_called()
