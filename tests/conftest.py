from dataclasses import replace
import pytest
from log_runs import WhisperConfig


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
