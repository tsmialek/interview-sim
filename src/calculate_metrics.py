from typing import Tuple
import re
import jiwer

from log_runs import Artifact, RunResults

TERM_PATTERN = re.compile(r"\{([^|]+)\|([^}]+)\}")
cer_wer_normalizer = jiwer.Compose(
    [
        jiwer.SubstituteRegexes({r"\b\w+-(?!\w)\s*": ""}),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveSpecificWords(["yyy", "eee", "mmm", "e", "y"]),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def calculate_metrics(
    run_artifact: Artifact,
    parsed_reference_text: str,
    reference_terms: list[Tuple[str, str]],
) -> RunResults:
    pass


def parse_reference(text: str) -> Tuple[str, list[Tuple[str, str]]]:
    terms = TERM_PATTERN.findall(text)
    final_text = TERM_PATTERN.sub(r"\1", text).strip()
    return (final_text, terms)


def normalize_for_cer_wer(text: str) -> str:
    return str(cer_wer_normalizer(text))


def normalize_for_terms(text: str) -> str:
    return re.sub(r'\.{3}|[,?!()\[\]{}"\'„”«»]|\.(?=\s|$)', "", text.lower().strip())
