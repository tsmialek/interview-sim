"""Read transcripts in the {form|term} format and normalize text for evaluation.

The same normalize() is applied to reference transcripts and to ASR output, so the word
positions of annotated terms line up with the words used for WER and CER.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TERM_RE = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")
# a run of a single filler letter, standing alone; Whisper also writes "hm" and "mhm"
FILLER_RE = re.compile(r"(?<![\w-])(y+|e+|m+|hm+|mhm)(?![\w-])")
DIGITS = "zero jeden dwa trzy cztery pięć sześć siedem osiem dziewięć".split()


def versions_to_words(text: str) -> str:
    """Spell out version numbers, as the transcripts do: "1.4.1" -> "jeden kropka cztery kropka jeden"."""
    return re.sub(
        r"\b\d(?:\.\d)+\b",
        lambda m: " kropka ".join(DIGITS[int(d)] for d in m.group().split(".")),
        text,
    )


def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation, apostrophes and fillers."""
    # the apostrophe is spelling only ("rollback'owaniem" = "rollbackowaniem")
    text = versions_to_words(text).lower().replace("'", "")
    # so are separators inside a word: "pg-vector" = "pgvector", "CI/CD" = "cicd", "Next.js" = "nextjs";
    # versions are spelled out above, so their dots are gone by now
    text = re.sub(r"(?<=\w)[-/.](?=\w)", "", text)
    words = re.sub(r"[^\w\s-]", " ", text).split()
    return [w for w in words if not FILLER_RE.fullmatch(w)]


def plain_text(text: str) -> str:
    """Keep only the spoken form of each term: "{Grafany|Grafana}" -> "Grafany"."""
    return TERM_RE.sub(r"\1", text)


def count_fillers(text: str) -> int:
    return len(FILLER_RE.findall(text.lower()))


def parse_transcript(text: str) -> tuple[list[str], list[dict]]:
    """Normalized words and the annotated terms with their word positions.

    Each term is {"term", "form", "start", "end"}, and words[start:end] is its spoken form.
    """
    words: list[str] = []
    terms: list[dict] = []
    position = 0
    for match in TERM_RE.finditer(text):
        words += normalize(text[position:match.start()])
        form, term = match.groups()
        form_words = normalize(form)
        if form_words:
            start = len(words)
            words += form_words
            terms.append({"term": term, "form": " ".join(form_words), "start": start, "end": len(words)})
        position = match.end()
    words += normalize(text[position:])
    return words, terms


def load_transcript(file_id: str) -> str:
    return (DATA_DIR / "transcripts" / f"{file_id}.txt").read_text(encoding="utf-8")


def load_term_names(categories: set[str] | None = None) -> list[str]:
    """Term names from the dictionary, optionally only those of the given categories."""
    with open(DATA_DIR / "terms_it.csv", encoding="utf-8", newline="") as f:
        return [row["term"] for row in csv.DictReader(f) if categories is None or row["category"] in categories]
