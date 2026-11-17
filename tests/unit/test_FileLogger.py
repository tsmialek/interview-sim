from datetime import datetime, timezone
import random

from log_runs import FileRunRepository, RunConfig


def test_new_run_id_returns_proper_run_id(tmp_path, mocker):
    mock_datetime = mocker.patch("log_runs.datetime")
    mock_datetime.now.return_value = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    random.seed(42)

    repo = FileRunRepository(tmp_path)
    assert repo._new_run_id() == "20260115-1200-Nbrn"


def test_generate_config_id_ommits_field_order(default_run_config, tmp_path, mocker):
    repo = FileRunRepository(tmp_path)
    expected_id = repo._generate_config_id(default_run_config)

    original_dict = RunConfig.model_dump(
        default_run_config, mode="json", exclude={"config_id", "created_at"}
    )
    reversed_dict = dict(reversed(list(original_dict.items())))
    mocker.patch.object(RunConfig, "model_dump", return_value=reversed_dict)

    assert repo._generate_config_id(default_run_config) == expected_id
