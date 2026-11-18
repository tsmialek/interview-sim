from typing import Iterator
from pathlib import Path
from faster_whisper import WhisperModel

from constants import AUDIO_DIR, KEY_PROJECT_PACKAGES, RECORDINGS_PATH
from log_runs import AudioSplit, RunConfig, RunRepository
from process import transcribe
from project_data import get_recordings_for_split


def run_variant_first(
    config: RunConfig,
    split: AudioSplit,
    repo: RunRepository,
    recordings_csv: Path = RECORDINGS_PATH,
    recordings_folder: Path = AUDIO_DIR,
    key_packages: list[str] = KEY_PROJECT_PACKAGES,
) -> str:
    config_id = repo.save_config(config)
    recordings = get_recordings_for_split(
        recordings_csv=recordings_csv,
        recordings_folder=recordings_folder,
        split=split,
    )

    whisper_model = WhisperModel(
        config.whisper.model,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
    )

    with repo.start_run(
        config_id=config_id, split=split, key_packages=key_packages
    ) as run:
        for rec in recordings:
            artifact = transcribe(whisper_model, config.whisper, rec)
            run.save_artifact(artifact)

        return run.run_id

    raise RuntimeError(
        "Returned run_id is empty. Something went wrong during run execution"
    )
