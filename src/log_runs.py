from __future__ import annotations

import time
from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Union, Self, Dict, Any, Literal, Annotated
from datetime import datetime, timezone
import random
import string
from pathlib import Path
from abc import ABC, abstractmethod
import json
import csv

from faster_whisper.transcribe import Word

from constants import PROJECT_DIR, RUNS_DIR
from utils import (
    get_git_changed_files,
    get_current_diffs,
    get_last_commit,
    get_project_environment,
    get_sha256,
)


class Run(BaseModel):
    run_id: str
    config_id: str
    source_run_id: str | None
    split: str
    rec_ids: List[str]
    started_at: str
    finished_at: str
    duration_s: float
    status: str
    error: str | None
    git_commit: str
    git_dirty: bool
    packages: Dict[str, str]


class SlmConfig(BaseModel):
    model: str
    digest: str
    prompt_sha: str
    temperature: float


class DictionaryConfig(BaseModel):
    terms_sha: str
    hidden_terms_sha: str
    rapidfuzz_threshold: float


class WhisperConfig(BaseModel):
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    temperature: Union[float, List[float]]
    vad_filter: bool
    word_timestamps: bool = True
    initial_prompt: str | None = None
    no_speech_threshold: float | None = None
    log_prob_threshold: float | None = None
    compression_ratio_threshold: float | None = None
    hotwords: List[str] | str | None = None
    hotwords_source: str | None = None

    def get_transcribe_args(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for field, value in self.model_dump(
            exclude={"model", "device", "compute_type", "hotwords_source"},
            exclude_none=True,
        ).items():
            if field == "hotwords" and isinstance(value, list):
                args[field] = " ".join(value)
            else:
                args[field] = value
        return args


class RunConfig(BaseModel):
    config_id: str | None = None
    variant: int
    parent_config_id: str | None = None
    created_at: datetime | None = None
    whisper: WhisperConfig
    dictionary: DictionaryConfig | None = None
    slm: SlmConfig | None = None


AudioSplit = Literal["dev", "test"]


class Replacement(BaseModel):
    r_from: str
    r_to: str
    base_term: str
    position: int
    source: str


class RunArtifact(BaseModel):
    rec_id: str
    final_text: str


class TranscribeArtifact(RunArtifact):
    kind: Literal["transcribe"] = "transcribe"
    words: List[Word]
    transcribe_s: float


class ReplaceArtifact(RunArtifact):
    kind: Literal["replace"] = "replace"
    replacements: List[Replacement]
    rejected: List[Replacement]
    replace_s: float


Artifact = Annotated[
    Union[TranscribeArtifact, ReplaceArtifact], Field(discriminator="kind")
]
artifact_adapter = TypeAdapter(Artifact)


class RunResults(BaseModel):
    run_id: str
    rec_id: str
    ref_chars: int
    char_errors: int
    ref_words: int
    word_errors: int
    terms_known: int
    hits_known: int
    form_hits_known: int
    terms_hidden: int
    hits_hidden: int
    form_hits_hidden: int
    false_replacements: int
    duration_s: float


class RunsLogger(ABC):
    @abstractmethod
    def save_config(self, run_confg: RunConfig) -> str:
        pass

    @abstractmethod
    def get_config(self, config_id: str) -> RunConfig | None:
        pass

    @abstractmethod
    def start_run(
        self,
        config_id: str,
        split: str,
        key_packages: list[str],
        source_run_id: str | None = None,
    ) -> ActiveRun:
        pass

    @abstractmethod
    def save_artifact(self, run_id: str, run_artifact: RunArtifact) -> None:
        pass

    @abstractmethod
    def get_artifact(self, run_id: str, rec_id: str) -> RunArtifact:
        pass

    @abstractmethod
    def save_run(self, run: Run) -> None:
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> Run:
        pass

    @abstractmethod
    def save_results(self, run_results: RunResults) -> None:
        pass

    @abstractmethod
    def get_results(self, run_id: str, rec_id: str | None) -> RunResults:
        pass

    @abstractmethod
    def save_code_patch(self, run_id: str, diff: str) -> None:
        pass


class ActiveRun:
    def __init__(
        self,
        logger: RunsLogger,
        run_id: str,
        config_id: str,
        split: str,
        source_run_id: str | None,
        key_packages: list[str],
        project_dir: Path | str = PROJECT_DIR,
    ) -> None:
        self.logger = logger
        self.run_id = run_id
        self.config_id = config_id
        self.split = split
        self.source_run_id = source_run_id

        self.start_time: float = 0.0
        self.end_time: float = 0.0

        self.error: str | None = None
        self.status: str | None = None
        self.rec_ids: List[str] = []
        self.commit: str = ""
        self.git_dirty: bool = False
        self.project_env: Dict[str, str] = {}
        self.started = datetime.now(tz=timezone.utc)
        self.finished = datetime.now(tz=timezone.utc)
        self.key_packages = key_packages
        self.project_dir = project_dir

    def __enter__(self) -> Self:
        self.started = datetime.now(tz=timezone.utc)
        self.start_time = time.perf_counter()

        self.project_env = get_project_environment(self.key_packages)
        self.commit = get_last_commit(self.project_dir)
        self.git_dirty = True if get_git_changed_files(self.project_dir) else False
        current_code_diff = get_current_diffs(self.project_dir)

        if current_code_diff.strip():
            self.logger.save_code_patch(self.run_id, current_code_diff)

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finished = datetime.now(tz=timezone.utc)
        self.end_time = time.perf_counter()
        self.status = "failed" if exc_type else "ok"
        self.error = repr(exc) if exc else None

        run = Run(
            run_id=self.run_id,
            config_id=self.config_id,
            source_run_id=self.source_run_id,
            split=self.split,
            rec_ids=self.rec_ids,
            started_at=self.started.isoformat(),
            finished_at=self.finished.isoformat(),
            duration_s=self.end_time - self.start_time,
            status=self.status,
            error=self.error,
            git_commit=self.commit,
            git_dirty=self.git_dirty,
            packages=self.project_env,
        )
        self.logger.save_run(run)
        return False

    def save_artifact(self, artifact: RunArtifact) -> None:
        self.rec_ids.append(artifact.rec_id)
        self.logger.save_artifact(self.run_id, artifact)

    def get_artifact(self, rec_id: str) -> RunArtifact:
        return self.logger.get_artifact(self.run_id, rec_id)


class FileLogger(RunsLogger):
    def __init__(self, runs_folder: Path | str = RUNS_DIR) -> None:
        if isinstance(runs_folder, str):
            runs_folder = Path(runs_folder)

        runs_folder.mkdir(parents=True, exist_ok=True)
        self.runs_folder = runs_folder
        self._configs_file_path = runs_folder / "configs.jsonl"
        self._runs_file_path = runs_folder / "runs.jsonl"
        self._results_file_path = runs_folder / "results.csv"

    def _new_run_id(self):
        now = datetime.now(timezone.utc)
        chars = string.ascii_letters + string.digits
        random_chars = "".join(random.choices(chars, k=4))
        return f"{now.year}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}-{random_chars}"

    def _generate_config_id(self, run_config: RunConfig) -> str:
        config_json_string = json.dumps(
            run_config.model_dump(mode="json", exclude={"config_id", "created_at"}),
            sort_keys=True,
        )

        return get_sha256(config_json_string, length=8)

    def save_config(self, run_config: RunConfig) -> str:
        config_id = self._generate_config_id(run_config)
        if self.get_config(config_id):
            return config_id

        to_save = run_config.model_copy(
            update={"config_id": config_id, "created_at": datetime.now(tz=timezone.utc)}
        )

        with open(self._configs_file_path, "a+", encoding="utf-8") as file:
            file.write(to_save.model_dump_json() + "\n")

        return config_id

    def get_config(self, config_id: str) -> RunConfig | None:
        if not self._configs_file_path.exists():
            return None

        with open(self._configs_file_path, "r", encoding="utf-8") as file:
            for line in file:
                config = RunConfig.model_validate_json(line)
                if config.config_id == config_id:
                    return config

        return None

    def start_run(
        self,
        config_id: str,
        split: str,
        key_packages: list[str],
        source_run_id: str | None = None,
    ) -> ActiveRun:
        config = self.get_config(config_id)
        if not config:
            raise ValueError(
                f"Could not find config with id: {config_id} in: {self._configs_file_path}"
            )

        run_id = self._new_run_id()
        (self.runs_folder / run_id).mkdir()
        return ActiveRun(
            logger=self,
            config_id=config_id,
            split=split,
            run_id=run_id,
            source_run_id=source_run_id,
            key_packages=key_packages,
        )

    def save_artifact(self, run_id: str, run_artifact: Artifact) -> None:
        with open(
            self.runs_folder / run_id / f"{run_artifact.rec_id}.json",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(artifact_adapter.dump_json(run_artifact).decode("utf-8") + "\n")

    def get_artifact(self, run_id: str, rec_id: str) -> Artifact:
        with open(
            self.runs_folder / run_id / f"{rec_id}.json", "r", encoding="utf-8"
        ) as file:
            return artifact_adapter.validate_json(file.read())

    def save_run(self, run: Run) -> None:
        with open(self._runs_file_path, "a", encoding="utf-8") as file:
            file.write(run.model_dump_json() + "\n")

    def get_run(self, run_id: str) -> Run | None:
        if not self._runs_file_path.exists():
            return None

        with open(self._runs_file_path, "r", encoding="utf-8") as file:
            for line in file:
                run = Run.model_validate_json(line)
                if run.run_id == run_id:
                    return run

    def save_code_patch(self, run_id: str, diff: str) -> None:
        with open(
            self.runs_folder / run_id / "code.patch", "w", encoding="utf-8"
        ) as file:
            file.write(diff + "\n")

    def save_results(self, run_results: RunResults) -> None:
        with open(self._results_file_path, "a+", encoding="utf-8") as file:
            result_dict = RunResults.model_dump(run_results)
            writer = csv.DictWriter(file, result_dict.keys())
            file.seek(0)
            line_count = sum(1 for _ in file)
            if line_count < 1:
                writer.writeheader()
            writer.writerow(result_dict)

    def get_results(self, run_id: str, rec_id: str | None = None) -> list[RunResults]:
        if not self._results_file_path.exists():
            return []

        results: list[RunResults] = []
        with open(self._results_file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for line in reader:
                result = RunResults.model_validate(line)
                if result.run_id == run_id:
                    if rec_id and result.rec_id == rec_id:
                        return [result]
                    elif not rec_id:
                        results.append(result)

        return results
