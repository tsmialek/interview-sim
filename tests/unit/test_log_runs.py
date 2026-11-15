import pytest
from dataclasses import replace

from log_runs import WhisperConfig


def test_get_transcribe_args_excludes_internal_fields(default_whisper_config):
    args = default_whisper_config.get_transcribe_args()

    for excluded in ["model", "device", "compute_type", "hotwords_source"]:
        assert excluded not in args


def test_get_transcribe_args_omits_none_values(default_whisper_config):
    cfg = WhisperConfig.model_copy(
        default_whisper_config,
        update={"initial_prompt": None, "no_speech_threshbold": None},
    )
    args = cfg.get_transcribe_args()

    assert "initial_prompt" not in args
    assert "no_speech_threshold" not in args


@pytest.mark.parametrize(
    "hotwords_input, expected_output",
    [
        (["słowo1", "słowo2"], "słowo1 słowo2"),
        ("gotowy string", "gotowy string"),
    ],
)
def test_get_transcribe_args_formats_hotwords_correctly(
    default_whisper_config, hotwords_input, expected_output
):
    cfg = WhisperConfig.model_copy(
        default_whisper_config, update={"hotwords": hotwords_input}
    )
    args = cfg.get_transcribe_args()

    assert args["hotwords"] == expected_output
