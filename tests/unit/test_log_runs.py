import pytest
from dataclasses import replace
import random
from unittest.mock import patch
from datetime import datetime, timezone
from log_runs import FileLogger


def test_get_transcribe_args_excludes_internal_fields(default_whisper_config):
    args = default_whisper_config.get_transcribe_args()

    for excluded in ["model", "device", "compute_type", "hotwords_source"]:
        assert excluded not in args


def test_get_transcribe_args_omits_none_values(default_whisper_config):
    cfg = replace(
        default_whisper_config, initial_prompt=None, no_speech_threshold=None
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
    cfg = replace(default_whisper_config, hotwords=hotwords_input)
    args = cfg.get_transcribe_args()

    assert args["hotwords"] == expected_output


@patch("log_runs.datetime")
def test_new_run_id_returns_proper_run_id(mock_datetime, tmp_path):
    mock_datetime.now.return_value = datetime(
        2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    random.seed(42)

    logger = FileLogger(tmp_path)
    assert logger._new_run_id() == "20260115-1200-Nbrn"
