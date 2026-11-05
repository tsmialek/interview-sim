"""Validate the D1 dataset: term dictionary, manifest, transcripts and audio.

Usage (from the repository root):
    python src/validate_data.py
    python src/validate_data.py --skip-audio   # e.g. before `dvc pull`

Exits with code 1 when any error is found. Warnings alone do not fail.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TERM_COLUMNS = ["term", "aliases", "category", "description"]
CATEGORIES = {"tool", "technology", "methodology", "concept", "acronym"}

MANIFEST_COLUMNS = [
    "file_id", "speaker", "sample_type", "question_id",
    "style", "device", "environment", "split",
]
MANIFEST_VALUES = {
    "sample_type": {"positive", "negative"},
    "style": {"casual", "prepared"},
    "device": {"phone", "laptop"},
    "environment": {"quiet", "noisy"},
    "split": {"dev", "test"},
}
CARDS = {
    "positive": {str(i) for i in range(1, 16)},
    "negative": {f"N{i}" for i in range(1, 7)},
}

ALLOWED_FILLERS = {"yyy", "eee", "mmm"}
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes, i.e. 16-bit PCM
MIN_DURATION, MAX_DURATION = 45.0, 90.0
FFMPEG_FIX = "ffmpeg -i data/audio/rec_XXX.wav -ar 16000 -ac 1 -c:a pcm_s16le tmp.wav && mv tmp.wav data/audio/rec_XXX.wav"

TERM_RE = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")
# a run of a single filler letter, standing alone (so "e-mail" or "mem" do not match)
FILLER_RE = re.compile(r"(?<![\w-])(y+|e+|m+)(?![\w-])")
DIGIT_RE = re.compile(r"\d+")


class Report:
    def __init__(self) -> None:
        self.issues: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def error(self, where: str, message: str) -> None:
        self.issues[where].append(("ERROR", message))

    def warn(self, where: str, message: str) -> None:
        self.issues[where].append(("WARN", message))

    def count(self, level: str) -> int:
        return sum(lvl == level for items in self.issues.values() for lvl, _ in items)

    def print(self) -> None:
        for where in sorted(self.issues):
            print(where)
            for level, message in self.issues[where]:
                print(f"  {level:<5}  {message}")
        print(f"\n{self.count('ERROR')} errors, {self.count('WARN')} warnings")


def context(text: str, start: int, end: int, width: int = 25) -> str:
    snippet = text[max(0, start - width):end + width].replace("\n", " ")
    return f"\"…{snippet}…\""


def read_csv(path: Path, columns: list[str], report: Report) -> list[tuple[str, dict]]:
    """Return (location, row) pairs for rows that have exactly the expected fields."""
    if not path.exists():
        report.error(path.name, "file not found")
        return []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            report.error(path.name, f"missing columns {missing}, found {reader.fieldnames}")
            return []
        rows = []
        for line, row in enumerate(reader, start=2):
            where = f"{path.name}:{line}"
            if None in row:
                report.error(where, "too many fields, quote values that contain a comma")
            elif any(row[c] is None for c in columns):
                report.error(where, "too few fields")
            else:
                rows.append((where, {k: v.strip() for k, v in row.items()}))
        return rows


def check_terms(path: Path, report: Report) -> dict[str, set[str]]:
    """Return each term mapped to its lowercase surface forms (term + aliases)."""
    forms: dict[str, set[str]] = {}
    alias_owner: dict[str, str] = {}
    for where, row in read_csv(path, TERM_COLUMNS, report):
        term = row["term"]
        if not term:
            report.error(where, "empty term")
            continue
        if term in forms:
            report.error(where, f"duplicate term '{term}'")
            continue
        if row["category"] not in CATEGORIES:
            report.error(where, f"unknown category '{row['category']}', allowed: {sorted(CATEGORIES)}")
        aliases = {a.strip().lower() for a in row["aliases"].split("|") if a.strip()}
        for alias in sorted(aliases):
            owner = alias_owner.setdefault(alias, term)
            if owner != term:
                report.warn(where, f"alias '{alias}' is also an alias of '{owner}'")
        forms[term] = {term.lower(), *aliases}
    return forms


def check_manifest(path: Path, report: Report) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for where, row in read_csv(path, MANIFEST_COLUMNS, report):
        file_id = row["file_id"]
        if file_id in rows:
            report.error(where, f"duplicate file_id '{file_id}'")
            continue
        for column, allowed in MANIFEST_VALUES.items():
            if row[column] not in allowed:
                report.error(where, f"{column} '{row[column]}' is not one of {sorted(allowed)}")
        cards = CARDS.get(row["sample_type"])
        if cards and row["question_id"] not in cards:
            expected = "1–15" if row["sample_type"] == "positive" else "N1–N6"
            report.error(where, f"question_id '{row['question_id']}' is not a {row['sample_type']} card ({expected})")
        rows[file_id] = row

    # both variants of one card from one speaker must stay in the same split
    splits = defaultdict(set)
    for row in rows.values():
        splits[(row["speaker"], row["question_id"])].add(row["split"])
    for (speaker, card), found in sorted(splits.items()):
        if len(found) > 1:
            report.error(path.name, f"speaker {speaker}, card {card} is in more than one split: {sorted(found)}")
    return rows


def check_transcript(path: Path, sample_type: str | None, forms: dict[str, set[str]],
                     patterns: dict[str, re.Pattern], report: Report) -> None:
    where = f"transcripts/{path.name}"
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        report.error(where, "empty transcript")
        return

    for match in TERM_RE.finditer(text):
        left, right = match.group(1), match.group(2)
        if not left.strip() or not right.strip():
            report.error(where, f"empty side in {match.group()}")
        elif right.strip() not in forms:
            hint = difflib.get_close_matches(right.strip(), forms, n=1)
            suffix = f", did you mean '{hint[0]}'?" if hint else ""
            report.error(where, f"'{right}' in {match.group()} is not a term in terms_it.csv{suffix}")
        elif left != left.strip() or right != right.strip():
            report.warn(where, f"extra spaces inside {match.group()}")
    if sample_type == "negative" and TERM_RE.search(text):
        report.error(where, "negative recording must not contain {form|term} markup")

    # everything below looks only at text outside the braces
    plain = TERM_RE.sub(" ", text)
    for char in "{}|":
        index = plain.find(char)
        if index != -1:
            report.error(where, f"unmatched '{char}', expected {{form|term}}: {context(plain, index, index + 1)}")

    lowered = plain.lower()
    fillers = [m for m in FILLER_RE.finditer(lowered) if m.group() not in ALLOWED_FILLERS]
    counts = Counter(m.group() for m in fillers)
    for filler, count in counts.items():
        first = next(m for m in fillers if m.group() == filler)
        report.error(where, f"filler '{filler}' ({count}×), use only yyy / eee / mmm: "
                            f"{context(plain, first.start(), first.end())}")

    for match in DIGIT_RE.finditer(plain):
        report.warn(where, f"digits '{match.group()}', write numbers as words: {context(plain, match.start(), match.end())}")

    if sample_type == "positive":
        seen = set()
        for term, pattern in patterns.items():
            for match in pattern.finditer(lowered):
                if (match.group(), term) not in seen:
                    seen.add((match.group(), term))
                    report.warn(where, f"'{match.group()}' looks like term '{term}' but has no braces: "
                                       f"{context(plain, match.start(), match.end())}")


def describe_header(header: bytes) -> str:
    if header[4:8] == b"ftyp":
        return "MP4/M4A container"
    if header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "MP3"
    if header[:4] == b"OggS":
        return "Ogg"
    if header[:4] == b"fLaC":
        return "FLAC"
    return "unknown format"


def check_audio(path: Path, report: Report) -> None:
    where = f"audio/{path.name}"
    with path.open("rb") as f:
        header = f.read(12)
    if not (header[:4] == b"RIFF" and header[8:12] == b"WAVE"):
        report.error(where, f"not a WAV file, actually {describe_header(header)}")
        return
    try:
        with wave.open(str(path)) as w:
            rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    except wave.Error as e:
        report.error(where, f"unreadable WAV: {e}")
        return

    problems = []
    if rate != SAMPLE_RATE:
        problems.append(f"{rate} Hz instead of {SAMPLE_RATE}")
    if channels != 1:
        problems.append(f"{channels} channels instead of mono")
    if width != SAMPLE_WIDTH:
        problems.append(f"{8 * width}-bit instead of {8 * SAMPLE_WIDTH}-bit")
    if problems:
        report.error(where, "wrong format: " + ", ".join(problems))
    duration = frames / rate
    if not MIN_DURATION <= duration <= MAX_DURATION:
        report.warn(where, f"duration {duration:.1f} s is outside {MIN_DURATION:.0f}–{MAX_DURATION:.0f} s")


def print_summary(manifest: dict[str, dict]) -> None:
    counts = Counter((row["split"], row["sample_type"]) for row in manifest.values())
    parts = []
    for split in sorted(MANIFEST_VALUES["split"]):
        positive, negative = counts[(split, "positive")], counts[(split, "negative")]
        parts.append(f"{split} {positive + negative} ({positive} positive, {negative} negative)")
    print(f"{len(manifest)} recordings: " + ", ".join(parts) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="default: data/ in the repository")
    parser.add_argument("--skip-audio", action="store_true", help="skip audio checks, e.g. before `dvc pull`")
    args = parser.parse_args(argv)

    report = Report()
    data_dir = args.data_dir
    forms = check_terms(data_dir / "terms_it.csv", report)
    manifest = check_manifest(data_dir / "recordings.csv", report)

    patterns = {
        term: re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(f, key=len, reverse=True))) + r")(?!\w)")
        for term, f in forms.items()
    }
    transcripts_dir = data_dir / "transcripts"
    transcripts = {p.stem: p for p in sorted(transcripts_dir.glob("*.txt"))}
    for file_id, path in transcripts.items():
        sample_type = manifest.get(file_id, {}).get("sample_type")
        check_transcript(path, sample_type, forms, patterns, report)

    audio_dir = data_dir / "audio"
    check_audio_files = not args.skip_audio
    if check_audio_files and not audio_dir.is_dir():
        report.error("audio/", "folder not found, run `dvc pull` or pass --skip-audio")
        check_audio_files = False
    audio = {}
    if check_audio_files:
        for path in sorted(audio_dir.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix != ".wav":
                report.error(f"audio/{path.name}", "unexpected file, recordings must be .wav")
                continue
            audio[path.stem] = path
            check_audio(path, report)

    sources = [("transcript", set(transcripts)), ("recordings.csv row", set(manifest))]
    if check_audio_files:
        sources.append(("audio", set(audio)))
    for file_id in sorted(set().union(*(ids for _, ids in sources))):
        for name, ids in sources:
            if file_id not in ids:
                report.error(file_id, f"missing {name}")

    print_summary(manifest)
    report.print()
    if any(where.startswith("audio/") and level == "ERROR"
           for where, items in report.issues.items() for level, _ in items):
        print(f"\nTo convert a recording to 16 kHz mono 16-bit WAV:\n  {FFMPEG_FIX}")
    return 1 if report.count("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
