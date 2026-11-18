import pandas as pd
from pathlib import Path

from log_runs import AudioSplit
from constants import DATA_DIR


class ProjectRepository:
    """Read-only access to project input data in data/ (versioned with DVC/git).

    Not to be confused with RunRepository (runs/, written by every run).
    Tests pass tmp_path as data_dir with a few fake files.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self._data_dir = data_dir
        self._recordings_csv = self._data_dir / "recordings.csv"
        self._recordings_folder = self._data_dir / "audio"
        self._transcripts_folder = self._data_dir / "transcripts"
        self._dictionary_csv = self._data_dir / "terms_it.csv"
        self._hidden_terms_txt = self._data_dir / "hidden_terms.txt"

    def get_recordings(self, split: AudioSplit) -> list[Path]:
        """Sorted .wav paths for `split`; raises ValueError if a listed file is missing."""
        return get_recordings_for_split(self._recordings_csv, self._recordings_folder, split)

    def get_reference(self, rec_id: str) -> str:
        """Raw reference transcript (with {form|term} braces) for `rec_id`."""
        raise NotImplementedError

    def get_hidden_terms(self) -> list[str]:
        """Hidden terms from data/hidden_terms.txt.

        Format: one canonical term per line, exactly as in the `term` column of
        terms_it.csv; empty lines are ignored.
        """
        raise NotImplementedError

    def get_dictionary(self, exclude_hidden: bool = True) -> pd.DataFrame:
        """terms_it.csv as a DataFrame: term, aliases (list[str], split on "|"),
        category, description. With exclude_hidden=True, hidden terms are removed.
        """
        raise NotImplementedError

    def get_dictionary_sha(self) -> str:
        """sha256 of terms_it.csv, for DictionaryConfig.terms_sha."""
        raise NotImplementedError

    def get_hidden_terms_sha(self) -> str:
        """sha256 of hidden_terms.txt, for DictionaryConfig.hidden_terms_sha."""
        raise NotImplementedError


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
