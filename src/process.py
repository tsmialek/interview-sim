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


def transcribe(
    whisper_model: WhisperModel, whisper_conf: WhisperConfig, audio_path: Path | str
) -> TranscribeArtifact:
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
    final_text = normalize_transcript(whisper_text)

    return TranscribeArtifact(
        rec_id=rec_id, final_text=final_text, whisper_text=whisper_text, words=words
    )


def replace_terms(
    generated_transcript: str,
    dictionary: pd.DataFrame,
    dict_config: DictionaryConfig,
    slm_config: SlmConfig | None,
) -> ReplaceArtifact:
    pass


def normalize_transcript(text: str) -> str:
    return text
