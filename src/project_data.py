import pandas as pd
from pathlib import Path

from log_runs import AudioSplit
from constants import DATA_DIR


class ProjectRepository:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self._data_dir = data_dir
        self._recordings_csv = self._data_dir / "recordings.csv"
        self._recordings_folder = self._data_dir / "audio"


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
