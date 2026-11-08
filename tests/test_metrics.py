from metrics import align, build_lexicon, find_term_spans, prf, term_counts
from transcripts import normalize, parse_transcript

LEXICON = build_lexicon(
    ["deploy", "Python", "Kafka", "sprint", "pull request", "REST", "feature", "merge request", "request"], ["Kafki"],
)


def counts(reference, hypothesis):
    words, terms = parse_transcript(reference)
    return term_counts(words, terms, normalize(hypothesis), LEXICON)


def test_identical_text_hits_every_term():
    result = counts("zrobiliśmy {deploy|deploy} i {pull request|pull request}", "Zrobiliśmy deploy i pull request.")
    assert result == {"tp": 2, "fn": 0, "fp": 0}


def test_misrecognized_term_is_a_miss():
    assert counts("zrobiliśmy {deploy|deploy} w piątek", "zrobiliśmy dplay w piątek") == {"tp": 0, "fn": 1, "fp": 0}


def test_dropped_term_is_a_miss():
    assert counts("zrobiliśmy {deploy|deploy} w piątek", "zrobiliśmy w piątek") == {"tp": 0, "fn": 1, "fp": 0}


def test_term_like_word_transcribed_as_spoken_is_not_false():
    # a story about running: "sprint" is in the dictionary, but it was really said
    assert counts("przygotowuję się do sprintu", "przygotowuję się do sprintu") == {"tp": 0, "fn": 0, "fp": 0}


def test_word_changed_into_term_is_false():
    assert counts("widziałem dużego pytona", "widziałem dużego Pythona") == {"tp": 0, "fn": 0, "fp": 1}


def test_wrong_form_of_real_term_is_only_a_miss():
    assert counts("nie mamy {Kafki|Kafka}", "nie mamy Kafka") == {"tp": 0, "fn": 1, "fp": 0}


def test_inserted_term_is_false():
    assert counts("zrobiliśmy to w piątek", "zrobiliśmy deploy to w piątek")["fp"] == 1


# the next two come from rec_001 and rec_003: the alignment puts the attempt beside the missed term


def test_attempt_substituted_next_to_missed_term_is_not_false():
    result = counts("o nowe nowe {featury|feature} nowe funkcjonalności", "o nowe featurey nowe funkcjonalności")
    assert result == {"tp": 0, "fn": 1, "fp": 0}


def test_attempt_inserted_next_to_missed_term_is_not_false():
    result = counts("na {merge requeście|merge request} skończona", "na merge requestie jest skończona")
    assert result == {"tp": 0, "fn": 1, "fp": 0}


def test_longest_match_first():
    assert find_term_spans(["otwieram", "pull", "request"], LEXICON) == [(1, 3)]


def test_short_entries_do_not_match_as_prefix():
    assert find_term_spans(["restauracja"], LEXICON) == []


def test_align_marks_substitutions_and_insertions():
    ref_equal, hyp_equal, hyp_to_ref = align(["a", "b", "c"], ["a", "x", "c", "d"])
    assert ref_equal == [True, False, True]
    assert hyp_equal == [True, False, True, False]
    assert hyp_to_ref == [0, 1, 2, None]


def test_prf():
    assert prf(0, 0, 0) == (None, None, None)
    precision, recall, f1 = prf(3, 1, 1)
    assert (precision, recall) == (0.75, 0.75)
    assert abs(f1 - 0.75) < 1e-9
