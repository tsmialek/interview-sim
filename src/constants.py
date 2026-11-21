from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs"
AUDIO_DIR = DATA_DIR / "audio"
DICTIONARY_PATH = DATA_DIR / "terms_it.csv"
RECORDINGS_PATH = DATA_DIR / "recordings.csv"

KEY_PROJECT_PACKAGES = [
    "faster-whisper",
    "spacy",
    "av",
    "ctranslate2",
    "pydantic",
    "rapidfuzz",
]
