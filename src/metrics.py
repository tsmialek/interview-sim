"""Term metrics: how well an ASR transcript reproduces the annotated IT terms.

A term is a hit (TP) when the hypothesis has exactly the same normalized form at the aligned
position, and a miss (FN) otherwise. A false term (FP) is a term-like span the hypothesis
changed or inserted where the reference has no term. Words transcribed as spoken never count
as false, so "sprintu" in a story about running is not an error. Neither do changed words right
next to a missed term: those are the same attempt, which the alignment placed beside the term.
"""

from __future__ import annotations

import jiwer

from transcripts import normalize

MIN_PREFIX_LEN = 5  # shorter entries would match ordinary words ("rest" -> "restauracja")


def align(ref_words: list[str], hyp_words: list[str]) -> tuple[list[bool], list[bool], list[int | None]]:
    """Word alignment as three lists.

    ref_equal[i]: reference word i was transcribed exactly
    hyp_equal[j]: hypothesis word j equals the reference word under it
    hyp_to_ref[j]: index of the reference word under hypothesis word j, None for insertions
    """
    ref_equal = [False] * len(ref_words)
    hyp_equal = [False] * len(hyp_words)
    hyp_to_ref: list[int | None] = [None] * len(hyp_words)
    if not ref_words or not hyp_words:  # jiwer rejects empty input
        return ref_equal, hyp_equal, hyp_to_ref

    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    for chunk in output.alignments[0]:
        if chunk.type not in ("equal", "substitute"):
            continue
        for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
            i, j = chunk.ref_start_idx + k, chunk.hyp_start_idx + k
            hyp_to_ref[j] = i
            if chunk.type == "equal":
                ref_equal[i] = hyp_equal[j] = True
    return ref_equal, hyp_equal, hyp_to_ref


def build_lexicon(term_names: list[str], forms: list[str]) -> set[tuple[str, ...]]:
    """Normalized word sequences that look like a term: dictionary names and annotated forms."""
    lexicon = set()
    for text in [*term_names, *forms]:
        words = normalize(text)
        if words:
            lexicon.add(tuple(words))
    return lexicon


def find_term_spans(hyp_words: list[str], lexicon: set[tuple[str, ...]]) -> list[tuple[int, int]]:
    """Term-like word spans in a hypothesis, longest match first ("pull request" is one term)."""
    max_len = max((len(entry) for entry in lexicon), default=0)
    # catches inflected forms never seen in the transcripts, e.g. "pythona" from "python"
    prefixes = tuple(entry[0] for entry in lexicon if len(entry) == 1 and len(entry[0]) >= MIN_PREFIX_LEN)

    spans = []
    j = 0
    while j < len(hyp_words):
        for n in range(min(max_len, len(hyp_words) - j), 0, -1):
            chunk = tuple(hyp_words[j:j + n])
            if chunk in lexicon or (n == 1 and chunk[0].startswith(prefixes)):
                spans.append((j, j + n))
                j += n
                break
        else:  # no entry starts here
            j += 1
    return spans


def _neighbourhood(j: int, hyp_to_ref: list[int | None], ref_len: int) -> range:
    """Reference words around hypothesis word j, from the aligned word before it to the one after.

    This includes the word under j and any reference words deleted next to it.
    """
    left = next((hyp_to_ref[k] for k in range(j - 1, -1, -1) if hyp_to_ref[k] is not None), 0)
    right = next((hyp_to_ref[k] for k in range(j + 1, len(hyp_to_ref)) if hyp_to_ref[k] is not None), ref_len - 1)
    return range(left, right + 1)


def match_terms(ref_words: list[str], terms: list[dict], hyp_words: list[str],
                lexicon: set[tuple[str, ...]]) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Hit terms, missed terms and false term spans (hypothesis word ranges) for one recording.

    terms come from parse_transcript().
    """
    ref_equal, hyp_equal, hyp_to_ref = align(ref_words, hyp_words)
    hits = [t for t in terms if all(ref_equal[t["start"]:t["end"]])]
    missed = [t for t in terms if not all(ref_equal[t["start"]:t["end"]])]

    in_term = {i for t in terms for i in range(t["start"], t["end"])}
    in_missed = {i for t in missed for i in range(t["start"], t["end"])}
    false = []
    for start, end in find_term_spans(hyp_words, lexicon):
        if all(hyp_equal[start:end]):
            continue  # transcribed as spoken
        if any(hyp_to_ref[j] in in_term for j in range(start, end)):
            continue  # an attempt at a real term, already counted as a hit or a miss
        if any(i in in_missed for j in range(start, end) for i in _neighbourhood(j, hyp_to_ref, len(ref_words))):
            continue  # the same attempt, aligned beside the missed term instead of under it
        false.append((start, end))
    return hits, missed, false


def term_counts(ref_words: list[str], terms: list[dict], hyp_words: list[str],
                lexicon: set[tuple[str, ...]]) -> dict[str, int]:
    """TP, FN and FP of terms for one recording; terms come from parse_transcript()."""
    hits, missed, false = match_terms(ref_words, terms, hyp_words, lexicon)
    return {"tp": len(hits), "fn": len(missed), "fp": len(false)}


def prf(tp: int, fn: int, fp: int) -> tuple[float | None, float | None, float | None]:
    """Precision, recall and F1, or None where undefined (e.g. recall of a recording without terms).

    For a whole split, sum the counts of all recordings first, then call this once.
    """
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        return precision, recall, None
    return precision, recall, 2 * precision * recall / (precision + recall)
