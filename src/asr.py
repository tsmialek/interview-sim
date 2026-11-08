"""Transcribe recordings with faster-whisper and read the saved results back.

Each run is saved as one JSON per recording in results/asr/<model>__<variant>/, so a change
in the evaluation or in the hallucination filter never needs a new transcription.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

from transcripts import DATA_DIR, load_term_names

RESULTS_DIR = DATA_DIR.parent / "results" / "asr"

BASE_PARAMS = {"language": "pl", "beam_size": 5, "vad_filter": True, "word_timestamps": True}
# written verbatim, with fillers: Whisper copies the style and stops smoothing the speech
VERBATIM_PROMPT = "Yyy, no więc, eee, pracowałem nad tym, mmm, projektem i, yyy, no, to było, eee, dość trudne."

KNOWN_HALLUCINATIONS = ("za oglądanie", "napisy wykonane", "napisy stworzone", "subskrybuj")
MAX_WORDS_PER_SECOND = 8  # people say 2-4 words per second


def load_recordings() -> list[dict]:
    with open(DATA_DIR / "recordings.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def hotwords(categories: set[str] | None = None) -> str:
    """Dictionary terms for the `hotwords` parameter, optionally only of the given categories.

    Joined with spaces: Whisper keeps at most 223 hotword tokens, and with commas
    all 94 terms take 266, so the end of the list would be silently cut off.
    """
    return " ".join(load_term_names(categories))


def transcribe_file(model: WhisperModel, model_name: str, variant: str, file_id: str, params: dict) -> dict:
    start_time = time.perf_counter()
    segments, info = model.transcribe(str(DATA_DIR / "audio" / f"{file_id}.wav"), **params)

    segs = []
    for seg in segments:  # the transcription only happens while iterating
        segs.append({
            "start": seg.start, "end": seg.end, "text": seg.text,
            "no_speech_prob": seg.no_speech_prob,
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,  # high means a loop
            "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                      for w in seg.words or []],
        })

    return {
        "id": file_id,
        "variant": variant,
        "model": model_name,
        "params": params,
        "duration": info.duration,
        "processing_time": time.perf_counter() - start_time,
        "segments": segs,
    }


def run(model_name: str, variant: str, params: dict, file_ids: list[str], results_dir: Path = RESULTS_DIR) -> Path:
    """Transcribe file_ids into results_dir/<model>__<variant>/ and return that folder.

    Files already there are skipped, and the model is loaded only when something is left to do.
    """
    out_dir = results_dir / f"{model_name}__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [file_id for file_id in file_ids if not (out_dir / f"{file_id}.json").exists()]
    if not todo:
        return out_dir

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    for file_id in todo:
        print(f"{out_dir.name} {file_id}…", end=" ", flush=True)
        result = transcribe_file(model, model_name, variant, file_id, params)
        (out_dir / f"{file_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{result['duration']:.0f} s of audio in {result['processing_time']:.0f} s")
    del model
    gc.collect()  # free the memory before the next, possibly larger, model
    return out_dir


def load_result(out_dir: Path, file_id: str) -> dict:
    return json.loads((out_dir / f"{file_id}.json").read_text(encoding="utf-8"))


def is_hallucination(seg: dict, min_logprob: float | None = None) -> bool:
    """A segment Whisper made up: impossibly fast, a known subtitle phrase, or (optionally) low confidence."""
    text = seg["text"].strip().lower()
    duration = max(seg["end"] - seg["start"], 0.01)
    if len(text.split()) / duration > MAX_WORDS_PER_SECOND:
        return True
    if any(phrase in text for phrase in KNOWN_HALLUCINATIONS):
        return True
    return min_logprob is not None and seg["avg_logprob"] < min_logprob


def clean_text(result: dict, use_filter: bool = True, min_logprob: float | None = None) -> str:
    segments = result["segments"]
    if use_filter:
        segments = [s for s in segments if not is_hallucination(s, min_logprob)]
    return " ".join(s["text"].strip() for s in segments)
