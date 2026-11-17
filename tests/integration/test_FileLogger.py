from datetime import datetime, timezone

from log_runs import (
    FileRunRepository,
    ReplaceArtifact,
    RunConfig,
    TranscribeArtifact,
    Run,
)
from tests.helpers import make_replacement, make_word

excluded_fields = {"config_id", "created_at"}


def test_save_config_empty_folder_creates_file(tmp_path, default_run_config):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    repo.save_config(default_run_config)

    configs_file = runs_dir / "configs.jsonl"
    assert len(configs_file.read_text(encoding="utf-8").splitlines()) == 1


def test_save_config_saves_config(tmp_path, default_run_config):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    configs_file = runs_dir / "configs.jsonl"
    configs_file.touch()

    before_save = datetime.now(tz=timezone.utc)
    config_id = repo.save_config(default_run_config)
    after_save = datetime.now(tz=timezone.utc)

    for line in configs_file.read_text(encoding="utf-8").splitlines():
        saved_config = RunConfig.model_validate_json(line)

    assert saved_config.config_id == config_id
    assert saved_config.created_at
    assert before_save <= saved_config.created_at <= after_save
    assert saved_config.model_dump(
        exclude=excluded_fields
    ) == default_run_config.model_dump(exclude=excluded_fields)


def test_save_run(tmp_path, get_default_run):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    repo.save_run(get_default_run)

    for run in (runs_dir / "runs.jsonl").read_text().splitlines():
        assert Run.model_validate_json(run) == get_default_run


def test_get_run(tmp_path, get_default_run):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    repo.save_run(get_default_run)

    assert repo.get_run(get_default_run.run_id) == get_default_run


def test_get_non_existent_run_returns_none(tmp_path):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    assert repo.get_run("123abc") == None


def test_save_config_two_identical_configs_are_saved_once(tmp_path, default_run_config):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    first_conifg_id = repo.save_config(default_run_config)
    second_config_id = repo.save_config(default_run_config)

    configs_file = runs_dir / "configs.jsonl"

    assert first_conifg_id == second_config_id
    assert len(configs_file.read_text(encoding="utf-8").splitlines()) == 1


def test_save_config_two_different_saved_separately(
    tmp_path, default_run_config, default_whisper_config
):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    first_conifg_id = repo.save_config(default_run_config)

    another_whisper_config = default_whisper_config.model_copy(
        update={"hotwords": ["test", "file", "repo"]}
    )
    another_config = default_run_config.model_copy(
        update={"whisper": another_whisper_config}
    )

    before_save = datetime.now(tz=timezone.utc)
    second_config_id = repo.save_config(another_config)
    after_save = datetime.now(tz=timezone.utc)
    configs_file = runs_dir / "configs.jsonl"

    config_lines = configs_file.read_text(encoding="utf-8").splitlines()
    assert first_conifg_id != second_config_id
    assert len(config_lines) == 2

    fetched_second_config = RunConfig.model_validate_json(config_lines[1])
    assert second_config_id == fetched_second_config.config_id
    assert fetched_second_config.created_at
    assert before_save <= fetched_second_config.created_at <= after_save

    fetched_second_config_user_fields = fetched_second_config.model_dump(
        exclude=excluded_fields
    )
    another_config_user_fields = another_config.model_dump(exclude=excluded_fields)
    assert fetched_second_config_user_fields == another_config_user_fields


def test_get_config_reads_config_by_id_properly(
    tmp_path, default_run_config, default_whisper_config
):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)

    another_whisper_config = default_whisper_config.model_copy(
        update={"hotwords": ["test", "file", "repo"]}
    )
    another_config = default_run_config.model_copy(
        update={"whisper": another_whisper_config}
    )

    before_save = datetime.now(tz=timezone.utc)
    second_config_id = repo.save_config(another_config)
    after_save = datetime.now(tz=timezone.utc)

    fetched_second_config = repo.get_config(second_config_id)

    assert fetched_second_config is not None
    assert second_config_id == fetched_second_config.config_id
    assert fetched_second_config.created_at
    assert before_save <= fetched_second_config.created_at <= after_save

    excluded_fields = {"config_id", "created_at"}
    fetched_second_config_user_fields = fetched_second_config.model_dump(
        exclude=excluded_fields
    )
    another_config_user_fields = another_config.model_dump(exclude=excluded_fields)
    assert fetched_second_config_user_fields == another_config_user_fields


def test_save_config_does_not_mutate_input(tmp_path, default_run_config):
    repo = FileRunRepository(tmp_path / "runs")
    original = default_run_config.model_copy(deep=True)

    repo.save_config(default_run_config)

    assert default_run_config == original


def test_save_and_get_artifact_returns_proper_instance(tmp_path):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    run_id = repo._new_run_id()
    (runs_dir / run_id).mkdir(exist_ok=True)

    whisper_text = "This is whisper text"
    words = [make_word(w) for w in whisper_text.split()]
    t_rec_id = "rec_t"
    r_rec_id = "rec_r"

    t_art = TranscribeArtifact(
        rec_id=t_rec_id,
        final_text="This is final text",
        words=words,
        transcribe_s=10.0,
    )

    r_art = ReplaceArtifact(
        rec_id=r_rec_id,
        final_text="This is final text replacement",
        rejected=[],
        replacements=[
            make_replacement("fial", "final"),
            make_replacement("replejsment", "replacement"),
        ],
        replace_s=10.0,
    )

    repo.save_artifact(run_id, t_art)
    repo.save_artifact(run_id, r_art)

    current_run_dir = tmp_path / "runs" / run_id
    assert len(list(current_run_dir.iterdir())) == 2

    fetched_t_art = repo.get_artifact(run_id, t_rec_id)
    assert isinstance(fetched_t_art, TranscribeArtifact)
    assert t_art == fetched_t_art

    fetched_r_art = repo.get_artifact(run_id, r_rec_id)
    assert isinstance(fetched_r_art, ReplaceArtifact)
    assert r_art == fetched_r_art


def test_save_results_twice_saves_header_once(tmp_path, make_run_results):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    run_id = repo._new_run_id()

    first_result = make_run_results(run_id=run_id, rec_id="rec_1")
    second_result = make_run_results(run_id=run_id, rec_id="rec_2")

    repo.save_results(first_result)
    repo.save_results(second_result)

    results_file = runs_dir / "results.csv"

    result_lines = results_file.read_text().splitlines()
    assert len(result_lines) == 3

    for key in ["run_id", "rec_id", "ref_chars"]:
        assert key in result_lines[0]
    assert "rec_1" in result_lines[1]
    assert "rec_2" in result_lines[2]


def test_get_results_non_existend_rec_id_returns_empty_list(tmp_path, make_run_results):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    run_id = repo._new_run_id()

    first_result = make_run_results(run_id=run_id, rec_id="rec_1")
    second_result = make_run_results(run_id=run_id, rec_id="rec_2")

    repo.save_results(first_result)
    repo.save_results(second_result)

    assert repo.get_results(run_id, "rec_99") == []


def test_get_results_with_run_id_only_returns_list_of_all_records(
    tmp_path, make_run_results
):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    run_id = repo._new_run_id()

    first_result = make_run_results(run_id=run_id, rec_id="rec_1")
    second_result = make_run_results(run_id=run_id, rec_id="rec_2")

    repo.save_results(first_result)
    repo.save_results(second_result)

    assert repo.get_results(run_id) == [first_result, second_result]


def test_get_results_with_rec_id_returns_only_given_record_result(
    tmp_path, make_run_results
):
    runs_dir = tmp_path / "runs"
    repo = FileRunRepository(runs_dir)
    run_id = repo._new_run_id()

    first_result = make_run_results(run_id=run_id, rec_id="rec_1")
    second_result = make_run_results(run_id=run_id, rec_id="rec_2")

    repo.save_results(first_result)
    repo.save_results(second_result)

    assert repo.get_results(run_id, "rec_2") == [second_result]
