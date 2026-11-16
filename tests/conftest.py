from dataclasses import replace
import pytest
from log_runs import (
    Run,
    RunConfig,
    RunResults,
    WhisperConfig,
    SlmConfig,
    DictionaryConfig,
)
import datetime


@pytest.fixture
def default_whisper_config() -> WhisperConfig:
    return WhisperConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
        language="pl",
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        word_timestamps=False,
        initial_prompt=None,
        no_speech_threshold=None,
        log_prob_threshold=None,
        compression_ratio_threshold=None,
        hotwords=None,
        hotwords_source=None,
    )


@pytest.fixture
def default_slm_config() -> SlmConfig:
    return SlmConfig(
        model="bielik-11b", digest="123#abc", prompt_sha="abc#123", temperature=0.5
    )


@pytest.fixture
def default_dictionary_config() -> DictionaryConfig:
    return DictionaryConfig(
        terms_sha="123#123",
        hidden_terms_sha="",
        rapidfuzz_threshold=0.7,
    )


@pytest.fixture
def default_run_config(
    default_whisper_config, default_slm_config, default_dictionary_config
) -> RunConfig:
    return RunConfig(
        config_id="123#123#",
        variant=4,
        parent_config_id="321#321",
        created_at=datetime.datetime(2026, 11, 11, 16),
        dictionary=default_dictionary_config,
        slm=default_slm_config,
        whisper=default_whisper_config,
    )


@pytest.fixture
def make_run_results():
    def _factory(run_id: str, rec_id: str) -> RunResults:
        return RunResults(
            run_id=run_id,
            char_errors=10,
            duration_s=100,
            false_replacements=2,
            form_hits_hidden=11,
            form_hits_known=11,
            hits_hidden=3,
            hits_known=10,
            rec_id=rec_id,
            ref_chars=100,
            ref_words=20,
            terms_hidden=3,
            terms_known=10,
            word_errors=2,
        )

    return _factory


@pytest.fixture
def get_default_run():
    return Run(
        run_id="123@abc",
        config_id="abc@123",
        duration_s=20.5,
        error=None,
        finished_at=datetime.datetime(2026, 11, 11).isoformat(),
        git_commit="123hash321",
        git_dirty=False,
        packages={"faster-whisper": "1.1.1", "jiwer": "1.1b"},
        rec_ids=["rec_1", "rec_2"],
        source_run_id=None,
        split="dev",
        started_at=datetime.datetime(2026, 11, 12).isoformat(),
        status="ok",
    )
