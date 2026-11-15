from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RUNS_DIR = PROJECT_DIR / "runs"
AUDIO_DIR = PROJECT_DIR / "data" / "audio"
DICTIONARY_PATH = PROJECT_DIR / "data" / "terms_it.csv"
RECORDINGS_PATH = PROJECT_DIR / "data" / "recordings.csv"

KEY_PROJECT_PACKAGES = [
    "faster-whisper",
    "spacy",
    "av",
    "ctranslate2",
    "pydantic",
]
