from typing import Iterator
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel

from constants import AUDIO_DIR, KEY_PROJECT_PACKAGES, RECORDINGS_PATH
from log_runs import AudioSplit, RunConfig, RunsLogger
from process import transcribe


def run_variant_first(
    config: RunConfig,
    split: AudioSplit,
    logger: RunsLogger,
    recordings_csv: Path = RECORDINGS_PATH,
    recordings_folder: Path = AUDIO_DIR,
    key_packages: list[str] = KEY_PROJECT_PACKAGES,
) -> str:
    config_id = logger.save_config(config)
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

    with logger.start_run(
        config_id=config_id, split=split, key_packages=key_packages
    ) as run:
        for rec in recordings:
            artifact = transcribe(whisper_model, config.whisper, rec)
            run.save_artifact(artifact)

        return run.run_id

    raise RuntimeError(
        "Returned run_id is empty. Something went wrong during run execution"
    )


def get_recordings_for_split(
    recordings_csv: Path | str,
    recordings_folder: Path | str,
    split: AudioSplit,
) -> list[Path]:
    recordings_folder = Path(recordings_folder)
    recordings = pd.read_csv(recordings_csv, encoding="utf-8")
    valid_ids = sorted(
        set(recordings.loc[recordings["split"] == split, "file_id"].astype(str))
    )

    result: list[Path] = []
    for id in valid_ids:
        rec_path = recordings_folder / f"{id}.wav"
        if rec_path.exists():
            result.append(rec_path)
        else:
            raise ValueError(
                f"Missing {rec_path} recording thtat is present in {recordings_csv}"
            )

    return result
