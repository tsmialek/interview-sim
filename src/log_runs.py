from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple, Union, Self, Dict, Any
from datetime import datetime, timezone
import random
import string
from pathlib import Path
from abc import ABC, abstractmethod

from faster_whisper.transcribe import Word

from constants import DICTIONARY_PATH, RUNS_DIR
from utils import get_last_commit_hash, get_project_environment, is_worktree_dirty


@dataclass
class Run():
    run_id: str
    config_id: str
    source_run_id: str
    split: str
    rec_ids: List[str]
    started_at: datetime
    finished_at: datetime
    duration_s: float
    status: str
    error: Exception | None
    git_commit: str
    git_dirty: bool
    packages: Dict[str, str]


@dataclass
class SlmConfig():
    model: str
    digest: str
    prompt_sha: str
    temperature: float


@dataclass
class DictionaryConfig():
    terms_sha: str
    hidden_terms_sha: str
    rapidfuzz_threshold: float


@dataclass
class WhisperConfig():
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
        for field, value in self.__dict__.items():
            if field in ["model", "device", "compute_type", "hotwords_source"] or value is None:
                continue
            elif field == "hotwords" and isinstance(value, List):
                args[field] = " ".join(value)
            else:
                args[field] = value
        return args


@dataclass
class RunConfig():
    config_id: str
    variant: int
    parent_config_id: str | None
    created_at: datetime
    whisper: WhisperConfig
    dictionary: DictionaryConfig | None
    slm: SlmConfig | None


@dataclass
class Replacement():
    r_from: str
    r_to: str
    base_term: str
    position: int
    source: str


@dataclass
class RunArtifact():
    rec_id: str
    final_text: str


@dataclass
class TranscribeArtifact(RunArtifact):
    whisper_text: str
    words: List[Word]


@dataclass
class ReplaceArtifact(RunArtifact):
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
    def save_code_patch(self, run_id: str, rec_id: str, diff: str) -> None:
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
        self.error: Exception | None = None
        self.status: str | None = None
        self.rec_ids: List[str]

    def __enter__(self) -> Self:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end_time = time.perf_counter()
        self.status = "failed" if exc_type else "ok"
        self.error = exc

        commit = get_last_commit_hash()
        git_dirty = is_worktree_dirty()
        project_env = get_project_environment()

        run = Run(
            self.run_id,
            self.config_id,
            self.source_run_id,
            self.split,
            self.rec_ids,
            datetime.fromtimestamp(self.start_time),
            datetime.fromtimestamp(self.end_time),
            self.end_time - self.end_time,
            self.status,
            self.error,
            commit,
            git_dirty,
            project_env
        )
        self.logger.save_run(run)
        return False

    def save_artifact(self, artifact: RunArtifact) -> None:
        self.rec_ids.append(artifact.rec_id)
        self.logger.save_artifact(self.run_id, artifact)

    def get_artifact(self, rec_id: str) -> RunArtifact:
        return self.logger.get_artifact(rec_id, self.run_id)


class FileLogger(RunsLogger):
    def __init__(self, runs_path: Path | str = RUNS_DIR) -> None:
        if isinstance(runs_path, str):
            runs_path = Path(runs_path)

        runs_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = runs_path

    def _new_run_id(self):
        now = datetime.now(timezone.utc)
        chars = string.ascii_letters + string.digits
        random_chars = "".join(random.choices(chars, k=4))
        return f"{now.year}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}-{random_chars}"
