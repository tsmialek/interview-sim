from dataclasses import dataclass


@dataclass
class TranscriptStats():
    ref_chars: int
    char_errors: int

    ref_words: int
    word_errors: int

    terms_known: int
    hits_known: int
    form_hits_known: int

    terms_hidden: int
    hits_hidden: int
    form_hits_hidden: int

    false_replacements: int
    duration_s: float


def get_transcript_stats(final_text: str, target_text: str) -> TranscriptStats:
    pass
