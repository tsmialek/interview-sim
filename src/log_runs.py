from __future__ import annotations

import time
from pydantic import BaseModel
from typing import List, Tuple, Union, Self, Dict, Any, Literal
from datetime import datetime, timezone
import random
import string
from pathlib import Path
from abc import ABC, abstractmethod
import json
import csv

from faster_whisper.transcribe import Word
import pandas as pd

from constants import PROJECT_DIR, RUNS_DIR
from utils import get_git_changed_files, get_current_diffs, get_last_commit, get_project_environment, get_sha256


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
    temperature: Union[float, List[float], Tuple[float, ...]]
    vad_filter: bool
    word_timestamps: bool
    initial_prompt: str | None
    no_speech_threshold: float | None
    log_prob_threshold: float | None
    compression_ratio_threshold: float | None
    hotwords: List[str] | str | None
    hotwords_source: str | None

    def get_transcribe_args(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for field, value in self.model_dump(
                exclude={"model", "device", "compute_type", "hotwords_source"},
                exclude_none=True).items():
            if field == "hotwords" and isinstance(value, list):
                args[field] = " ".join(value)
            else:
                args[field] = value
        return args


class RunConfig(BaseModel):
    config_id: str
    variant: int
    parent_config_id: str | None
    created_at: datetime
    whisper: WhisperConfig
    dictionary: DictionaryConfig | None
    slm: SlmConfig | None


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
    whisper_text: str
    words: List[Word]


class ReplaceArtifact(RunArtifact):
    kind: Literal["replace"] = "replace"
    replacements: List[Replacement]
    rejected: List[Replacement]


class RunsLogger(ABC):
    @abstractmethod
    def save_config(self, run_confg: RunConfig) -> str:
        pass

    @abstractmethod
    def get_config(self, config_id: str) -> RunConfig:
        pass

    @abstractmethod
    def start_run(self, config: RunConfig, split: str, source_run_id: str) -> ActiveRun:
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
    def save_results(self) -> None:
        pass

    @abstractmethod
    def save_code_patch(self, run_id: str, diff: str) -> None:
        pass


class ActiveRun:
    def __init__(self, logger: RunsLogger, run_id: str, config_id: str, split: str, source_run_id: str) -> None:
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

    def __enter__(self) -> Self:
        self.started = datetime.now(tz=timezone.utc)
        self.start_time = time.perf_counter()

        self.commit = get_last_commit(PROJECT_DIR)
        self.git_dirty = True if get_git_changed_files(PROJECT_DIR) else False
        self.project_env = get_project_environment()
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
            packages=self.project_env
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
        config_json_string = json.dumps(RunConfig.model_dump(
            run_config, mode="json", exclude={"config_id", "created_at"}), sort_keys=True,)

        return get_sha256(config_json_string, length=8)

    def save_config(self, run_config: RunConfig) -> str:
        config_id = self._generate_config_id(run_config)
        # TODO
        return config_id

    def get_config(self, config_id: str) -> RunConfig | None:
        pass

    def start_run(self, config: RunConfig, split: str, source_run_id: str) -> ActiveRun:
        return ActiveRun(
            logger=self,
            config_id=config.config_id,
            split=split,
            run_id=self._new_run_id(),
            source_run_id=source_run_id)

    def save_artifact(self, run_id: str, run_artifact: RunArtifact) -> None:
        raise NotImplementedError

    def get_artifact(self, run_id: str, rec_id: str) -> RunArtifact:
        raise NotImplementedError

    def save_run(self, run: Run) -> None:
        raise NotImplementedError

    def save_code_patch(self, run_id: str, diff: str) -> None:
        raise NotImplementedError

    def save_results(self) -> None:
        raise NotImplementedError
