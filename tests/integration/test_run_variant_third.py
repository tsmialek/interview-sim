import pandas as pd
import pytest

from log_runs import DictionaryConfig, FileRunRepository, RunConfig, TranscribeArtifact
from run_variant import run_variant_third

SOURCE_TEXTS = {
    "rec_1": "mamy taski w dżirze",
    "rec_2": "obraz dokera jest gotowy",
}


class FakeProject:
    """Stands in for ProjectRepository: a two-term dictionary and fixed hashes."""

    def __init__(self, hidden_terms=None):
        self.hidden_terms = hidden_terms or []

    def get_dictionary(self, exclude_hidden=True):
        dictionary = pd.DataFrame(
            {
                "term": ["Jira", "Docker"],
                "aliases": [["dżira", "dżirze"], ["doker", "dokera"]],
            }
        )
        if not exclude_hidden:
            return dictionary
        hidden = {term.lower() for term in self.hidden_terms}
        return dictionary[~dictionary["term"].str.lower().isin(hidden)]

    def get_hidden_terms(self):
        return self.hidden_terms

    def get_dictionary_sha(self):
        return "terms-sha"

    def get_hidden_terms_sha(self):
        return "hidden-sha"


@pytest.fixture
def repo(tmp_path):
    return FileRunRepository(tmp_path / "runs")


@pytest.fixture
def parent_whisper(default_whisper_config):
    return default_whisper_config.model_copy(update={"hotwords": ["Jira", "Docker"]})


@pytest.fixture
def parent_config_id(repo, parent_whisper):
    return repo.save_config(RunConfig(variant=2, whisper=parent_whisper))


@pytest.fixture
def source_run_id(repo, parent_config_id):
    """A finished variant 2 run on dev with one artifact per SOURCE_TEXTS entry."""
    with repo.start_run(config_id=parent_config_id, split="dev", key_packages=[]) as run:
        for rec_id, text in SOURCE_TEXTS.items():
            run.save_artifact(
                TranscribeArtifact(
                    rec_id=rec_id, final_text=text, words=[], transcribe_s=1.0
                )
            )
    return run.run_id


def make_config(parent_config_id, whisper, threshold=90):
    return RunConfig(
        variant=3,
        parent_config_id=parent_config_id,
        whisper=whisper,
        dictionary=DictionaryConfig(
            terms_sha="", hidden_terms_sha="", rapidfuzz_threshold=threshold
        ),
    )


def run_third(repo, config, project=None, split="dev"):
    return run_variant_third(
        config, split, repo, project or FakeProject(), key_packages=[]
    )


def read_runs_dir(repo):
    """Contents of configs.jsonl and runs.jsonl, to check that nothing was saved."""
    files = [repo.runs_folder / "configs.jsonl", repo.runs_folder / "runs.jsonl"]
    return [f.read_text(encoding="utf-8") if f.exists() else "" for f in files]


def test_corrects_every_recording_of_source_run(
    repo, parent_config_id, source_run_id, default_whisper_config
):
    run_id = run_third(repo, make_config(parent_config_id, default_whisper_config))

    first = repo.get_artifact(run_id, "rec_1")
    second = repo.get_artifact(run_id, "rec_2")
    assert first.kind == "replace"
    assert first.final_text == "mamy taski w Jira"
    assert [r.r_from for r in first.replacements] == ["dżirze"]
    assert first.rejected == []
    assert first.replace_s >= 0
    assert second.final_text == "obraz Docker jest gotowy"


def test_run_points_to_source_run(
    repo, parent_config_id, source_run_id, default_whisper_config
):
    run_id = run_third(repo, make_config(parent_config_id, default_whisper_config))

    run = repo.get_run(run_id)
    assert run.source_run_id == source_run_id
    assert run.status == "ok"
    assert run.rec_ids == list(SOURCE_TEXTS)


def test_fills_in_variant_whisper_and_dictionary_hashes(
    repo, parent_config_id, source_run_id, parent_whisper, default_whisper_config
):
    run_id = run_third(repo, make_config(parent_config_id, default_whisper_config))

    saved = repo.get_config(repo.get_run(run_id).config_id)
    assert saved.variant == 3
    assert saved.parent_config_id == parent_config_id
    assert saved.whisper == parent_whisper
    assert saved.dictionary.terms_sha == "terms-sha"
    assert saved.dictionary.hidden_terms_sha == "hidden-sha"
    assert saved.dictionary.rapidfuzz_threshold == 90


def test_raises_without_source_run_and_saves_nothing(
    repo, parent_config_id, source_run_id, default_whisper_config
):
    before = read_runs_dir(repo)

    with pytest.raises(ValueError, match="run variant 2 first"):
        run_third(
            repo, make_config(parent_config_id, default_whisper_config), split="test"
        )

    assert read_runs_dir(repo) == before


def test_raises_without_parent_config_id(repo, default_whisper_config):
    with pytest.raises(ValueError, match="parent_config_id"):
        run_third(repo, make_config(None, default_whisper_config))


def test_raises_for_threshold_on_0_1_scale(
    repo, parent_config_id, source_run_id, default_whisper_config
):
    config = make_config(parent_config_id, default_whisper_config, threshold=0.7)

    with pytest.raises(ValueError, match="0-100"):
        run_third(repo, config)


def test_raises_for_hidden_term_not_in_dictionary(
    repo, parent_config_id, source_run_id, default_whisper_config
):
    before = read_runs_dir(repo)

    with pytest.raises(ValueError, match="Jenkis"):
        run_third(
            repo,
            make_config(parent_config_id, default_whisper_config),
            project=FakeProject(hidden_terms=["Jenkis"]),
        )

    assert read_runs_dir(repo) == before
