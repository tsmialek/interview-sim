from dataclasses import replace
import pytest
from log_runs import RunConfig, WhisperConfig, SlmConfig, DictionaryConfig
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
def default_run_config(default_whisper_config) -> RunConfig:
    slm_config = SlmConfig(
        model="bielik-11b",
        digest="123#abc",
        prompt_sha="abc#123",
        temperature=0.5
    )

    dict_config = DictionaryConfig(
        terms_sha="123#123",
        hidden_terms_sha="",
        rapidfuzz_threshold=0.7,
    )
    return RunConfig(
        config_id="123#123#",
        variant=4,
        parent_config_id="321#321",
        created_at=datetime.datetime(2026, 11, 11, 16),
        dictionary=dict_config,
        slm=slm_config,
        whisper=default_whisper_config
    )
