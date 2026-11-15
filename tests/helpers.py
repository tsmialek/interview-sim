from types import SimpleNamespace

from faster_whisper.transcribe import Word

from log_runs import Replacement


def make_word(
    word: str, start: float = 0.0, end: float = 1.0, probability: float = 0.9
) -> Word:
    return Word(start, end, word, probability)


def make_fake_segment(
    text: str, start: float = 0.0, end: float = 1.0, words: list | None = None
):
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=words or [make_word(word) for word in text.strip().split(" ")],
    )


def make_replacement(w_from: str, w_to: str) -> Replacement:
    return Replacement(
        r_from=w_from,
        r_to=w_to,
        base_term=w_to,
        position=1,
        source="test",
    )
