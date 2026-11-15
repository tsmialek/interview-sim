from log_runs import (
    TranscribeArtifact,
    ReplaceArtifact,
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


def replace_terms(
    generated_transcript: str,
    dictionary: pd.DataFrame,
    dict_config: DictionaryConfig,
    slm_config: SlmConfig | None,
) -> ReplaceArtifact:
    pass
