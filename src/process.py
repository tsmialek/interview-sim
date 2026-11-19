from typing import Protocol

from log_runs import (
    TranscribeArtifact,
    Replacement,
    WhisperConfig,
    SlmConfig,
    DictionaryConfig,
)
from faster_whisper import WhisperModel
import pandas as pd
from pathlib import Path
import time


def transcribe(
    whisper_model: WhisperModel, whisper_conf: WhisperConfig, audio_path: Path | str
) -> TranscribeArtifact:
    start = time.perf_counter()
    segments, info = whisper_model.transcribe(
        audio=str(audio_path), **whisper_conf.get_transcribe_args()
    )

    rec_id = Path(audio_path).stem

    words = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if seg.words:
            words.extend(seg.words)

    whisper_text = " ".join(text_parts)
    final_text = whisper_text
    duration_s = time.perf_counter() - start

    return TranscribeArtifact(
        rec_id=rec_id,
        final_text=final_text,
        words=words,
        transcribe_s=duration_s,
    )

def build_hotwords(dictionary: pd.DataFrame, hidden_terms: list[str]) -> list[str]:
    """Return dictionary terms in CSV order, without the hidden ones.

    Terms are stripped; empty terms and duplicates are skipped.
    Raises ValueError if a hidden term is not in the dictionary
    (catches typos in hidden_terms.txt).
    """

    terms = dictionary["term"].str.strip().tolist()
    hidden_set = set(hidden_terms)
    missing = hidden_set - set(terms)
    if missing:
        raise ValueError(
            f"Hidden terms not in dictionary: {sorted(missing)}"
        )
    seen = set()
    hotwords = []

    for term in terms:
        if not term or term in hidden_set or term in seen:
            continue
        hotwords.append(term)
        seen.add(term)
    return hotwords

# Every correction step only PROPOSES replacements (list[Replacement]).
# apply_replacements is the only function that changes the text.
# None of these functions read files or know about run_id / RunRepository.

def find_dictionary_replacements(
    text: str,
    dictionary: pd.DataFrame,
    config: DictionaryConfig,
) -> list[Replacement]:
    """Variant 3 Find misrecognised dictionary terms in `text`.

    `dictionary`: ProjectRepository.get_dictionary(), hidden terms already removed.
    Returns replacements with source="dictionary"; `position` refers to `text`.
    """
    raise NotImplementedError


class SlmClient(Protocol):
    """Minimal interface to the local SLM, so tests can pass a fake client.

    A real implementation wraps ollama.chat(model=..., format=schema, options=...)
    and returns the raw response text (expected to be JSON matching `schema`).
    """

    def chat(self, system: str, user: str, schema: dict) -> str: ...


def find_slm_replacements(
    text: str,
    candidates: list[Replacement] | None,
    config: SlmConfig,
    client: SlmClient,
) -> list[Replacement]:
    """Variant 4 Ask the SLM for replacements, with Polish inflection.

    - candidates=None: the SLM sees the whole `text` and returns a JSON list of
      replacements (approach 1).
    - candidates given: the SLM gets each marked fragment (r_from at position) with
      its sentence and returns only the correct spelling for it (approach 2). Today
      the candidates come from the dictionary, in variant 5 from HerBERT.
    Returns RAW proposals with source="slm"; validation is done by
    validate_slm_replacements.
    """
    raise NotImplementedError


def validate_slm_replacements(
    text: str,
    proposals: list[Replacement],
) -> tuple[list[Replacement], list[Replacement]]:
    """Split SLM proposals into (accepted, rejected).

    Rules (poc-etap1-ustalenia.md, "Słownik i podmiana"):
    - text[position:position + len(r_from)] == r_from
    - r_to starts with base_term without its last 2 characters
      (Jira -> Jirze, Kafka -> Kafce); terms of up to 4 characters must appear
      in full (AWS -> AWS-ie)
    """
    raise NotImplementedError


def apply_replacements(text: str, replacements: list[Replacement]) -> str:
    """Return `text` with every replacement applied.

    Everything outside the replaced fragments must stay byte-for-byte unchanged.
    Apply from the highest position down, so earlier positions stay valid.
    Overlapping replacements are a conflict: decide and document the rule.
    """
    raise NotImplementedError
