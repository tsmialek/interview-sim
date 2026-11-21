from typing import Protocol
from faster_whisper.tokenizer import Tokenizer

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

def build_hotwords(dictionary: pd.DataFrame) -> list[str]:
    """Return dictionary terms in CSV order, to be passed as Whisper hotwords.

    Expects a dictionary with hidden terms already removed
    (ProjectRepository.get_dictionary does that). Terms are stripped;
    empty terms and duplicates are skipped.
    """

    terms = dictionary["term"].str.strip().tolist()
    seen = set()
    hotwords = []

    for term in terms:
        if not term or term in seen:
            continue
        hotwords.append(term)
        seen.add(term)
    return hotwords

def count_hotword_tokens(hotwords: list[str], tokenizer: Tokenizer) -> int:
    """Return the number of tokens Whisper sees for `hotwords`.

    Encodes the list exactly like faster-whisper: terms joined with spaces
    (WhisperConfig.get_transcribe_args) and a leading space (get_prompt).
    """
    text = " " + " ".join(hotwords)
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_dropped_hotwords(
    hotwords: list[str],
    tokenizer: Tokenizer,
    max_length: int = 448,
) -> list[str]:
    """Return the hotwords that faster-whisper cuts off from the prompt.

    get_prompt keeps at most max_length // 2 - 1 hotword tokens (223 for
    Whisper, whose max_length is 448). The first returned term may be cut
    partially; all later ones are dropped entirely. An empty list means
    the whole list fits.
    """
    limit = max_length // 2 - 1
    for i in range(len(hotwords)):
        prefix = hotwords[: i + 1]
        if count_hotword_tokens(prefix, tokenizer) > limit:
            return hotwords[i:]
    return []


def ensure_hotwords_fit(
    hotwords: list[str] | str | None,
    tokenizer: Tokenizer,
    max_length: int = 448,
) -> None:
    """Raise ValueError if faster-whisper would cut any hotwords from the prompt.

    None or empty hotwords pass. A str is checked as one block, so the error
    says that something is cut, not which terms.
    """
    if not hotwords:
        return
    if isinstance(hotwords, str):
        hotwords = [hotwords]
    dropped = get_dropped_hotwords(hotwords, tokenizer, max_length)
    if dropped:
        raise ValueError(
            f"{len(dropped)} hotwords would be cut by faster-whisper: {dropped}"
        )



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

    Everything outside the replaced fragments stays byte-for-byte unchanged.
    Replacements are applied from the highest position down, so earlier
    positions stay valid; the order of the input list does not matter.

    Raises ValueError before applying anything if `r_from` is not found
    in `text` at `position`, or if two replacements overlap (touching ones
    are fine). Resolving conflicts belongs to the step that proposes the
    replacements, so it can report the dropped ones in ReplaceArtifact.rejected.
    """

    ordered = sorted(replacements, key=lambda r: r.position)

    for r in ordered:
        end = r.position + len(r.r_from)

        if r.position < 0 or end > len(text):
            raise ValueError(
                f"Invalid replacement position: {r.position}"
            )

        if text[r.position:end] != r.r_from:
            raise ValueError(
                f"Expected {r.r_from!r} at position {r.position}"
            )

    for prev, nxt in zip(ordered, ordered[1:]):
        prev_end = prev.position + len(prev.r_from)

        if nxt.position < prev_end:
            raise ValueError(
                f"Overlapping replacements at "
                f"{prev.position} and {nxt.position}"
            )

    for r in reversed(ordered):
        start = r.position
        end = start + len(r.r_from)

        text = text[:start] + r.r_to + text[end:]

    return text

