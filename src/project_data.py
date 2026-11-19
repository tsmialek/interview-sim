import pandas as pd
from pathlib import Path

from log_runs import AudioSplit
from constants import DATA_DIR
from utils import get_sha256


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
        recordings_folder = Path(self._recordings_folder)
        recordings = pd.read_csv(self._recordings_csv, encoding="utf-8")
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
                    f"Missing {rec_path} recording thtat is present in {self._recordings_csv}"
                )

        return result

    def get_reference(self, rec_id: str) -> str:
        """Raw reference transcript (with {form|term} braces) for `rec_id`."""

        transcript_path = (self._transcripts_folder) / f"{rec_id}.txt"
        if transcript_path.exists():
            return transcript_path.read_text(encoding="utf-8")

        raise ValueError(
            f"Could not find reference transcript with id: {rec_id} in {self._transcripts_folder}"
        )

    def get_hidden_terms(self) -> list[str]:
        """Hidden terms from data/hidden_terms.txt.

        Format: one canonical term per line, exactly as in the `term` column of
        terms_it.csv; empty lines are ignored.
        """
        with open(self._hidden_terms_txt, "r", encoding="utf-8") as file:
            return [term.strip() for term in file.read().splitlines() if term.strip()]

    def get_dictionary(self, exclude_hidden: bool = True) -> pd.DataFrame:
        """terms_it.csv as a DataFrame: term, aliases (list[str], split on "|"),
        category, description. With exclude_hidden=True, hidden terms are removed.
        """
        terms_dict = pd.read_csv(self._dictionary_csv, encoding="utf-8")
        empty_terms = terms_dict["term"].isna()
        if empty_terms.any():
            raise ValueError(
                f"{empty_terms.sum()} lines have empty terms: {(terms_dict[empty_terms].index + 2).tolist()}"
            )

        empty_aliases = terms_dict["aliases"].isna()
        if empty_aliases.any():
            raise ValueError(
                f"{empty_aliases.sum()} terms have empty aliases: {terms_dict.loc[empty_aliases, 'term'].tolist()}"
            )

        terms_dict["aliases"] = terms_dict["aliases"].apply(
            lambda x: [v.strip() for v in x.split("|")]
        )
        if not exclude_hidden:
            return terms_dict
        hidden_terms = {term.lower() for term in self.get_hidden_terms()}

        return terms_dict[
            ~terms_dict["term"].str.strip().str.lower().isin(hidden_terms)
        ].reset_index(drop=True)

    def get_dictionary_sha(self) -> str:
        """sha256 of terms_it.csv, for DictionaryConfig.terms_sha."""
        with open(self._dictionary_csv, "r", encoding="utf-8") as file:
            dict_text = file.read()

        return get_sha256(dict_text)

    def get_hidden_terms_sha(self) -> str:
        """sha256 of hidden_terms.txt, for DictionaryConfig.hidden_terms_sha."""
        hidden_terms_txt = ",".join(sorted(self.get_hidden_terms()))

        return get_sha256(hidden_terms_txt)
