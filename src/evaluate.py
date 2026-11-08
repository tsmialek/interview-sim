"""Score saved ASR runs against the reference transcripts.

Scores are pooled over all recordings of a run (total errors / total words), not averaged
per recording, so longer recordings weigh more and recordings without terms do not break F1.
"""

from __future__ import annotations

from pathlib import Path

import jiwer

from asr import clean_text, load_recordings, load_result
from metrics import align, build_lexicon, match_terms, prf, term_counts
from transcripts import count_fillers, load_term_names, load_transcript, normalize, parse_transcript, plain_text


def term_lexicon() -> set[tuple[str, ...]]:
    """Term-like word sequences: dictionary names and the annotated forms of all recordings.

    The lexicon is part of the evaluation, not of the system, so using the test split here is fine.
    """
    forms = [t["form"] for rec in load_recordings() for t in parse_transcript(load_transcript(rec["file_id"]))[1]]
    return build_lexicon(load_term_names(), forms)


def score(out_dir: Path, file_ids: list[str], lexicon: set[tuple[str, ...]],
          use_filter: bool = True, min_logprob: float | None = None) -> dict:
    """WER, CER, term precision/recall/F1, fillers and speed of one run over file_ids.

    rtf is the real-time factor: seconds of processing per second of audio.
    """
    refs, hyps = [], []
    counts = {"tp": 0, "fn": 0, "fp": 0}
    fillers_gold = fillers = 0
    duration = processing_time = 0.0
    for file_id in file_ids:
        gold = load_transcript(file_id)
        words, terms = parse_transcript(gold)
        result = load_result(out_dir, file_id)
        text = clean_text(result, use_filter, min_logprob)
        hyp_words = normalize(text)

        refs.append(" ".join(words))
        hyps.append(" ".join(hyp_words))
        for key, value in term_counts(words, terms, hyp_words, lexicon).items():
            counts[key] += value
        fillers_gold += count_fillers(plain_text(gold))
        fillers += count_fillers(text)
        duration += result["duration"]
        processing_time += result["processing_time"]

    precision, recall, f1 = prf(**counts)
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
        "fillers_gold": fillers_gold,
        "fillers": fillers,
        "rtf": processing_time / duration,
    }


def term_errors(out_dir: Path, file_ids: list[str], lexicon: set[tuple[str, ...]],
                use_filter: bool = True, min_logprob: float | None = None) -> list[dict]:
    """Missed and false terms of one run, one row each: what was said and what Whisper wrote there."""
    rows = []
    for file_id in file_ids:
        words, terms = parse_transcript(load_transcript(file_id))
        hyp = normalize(clean_text(load_result(out_dir, file_id), use_filter, min_logprob))
        _, missed, false = match_terms(words, terms, hyp, lexicon)
        _, _, hyp_to_ref = align(words, hyp)
        ref_to_hyp = {i: j for j, i in enumerate(hyp_to_ref) if i is not None}

        for t in missed:
            written = " ".join(hyp[ref_to_hyp[i]] for i in range(t["start"], t["end"]) if i in ref_to_hyp)
            rows.append({"id": file_id, "error": "missed", "term": t["term"], "said": t["form"], "written": written or "∅"})
        for start, end in false:
            said = " ".join(words[hyp_to_ref[j]] for j in range(start, end) if hyp_to_ref[j] is not None)
            rows.append({"id": file_id, "error": "false", "term": None, "said": said or "∅",
                         "written": " ".join(hyp[start:end])})
    return rows
