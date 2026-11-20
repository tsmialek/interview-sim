from types import SimpleNamespace

import pytest

from log_runs import Replacement, TranscribeArtifact
from process import find_slm_replacements, transcribe, validate_slm_replacements
from tests.helpers import make_fake_segment, make_word

fake_info = SimpleNamespace(
    language="pl",
    language_probablity=0.95,
    duration=20.5,
    duration_after_vad=19,
    transcription_options=SimpleNamespace(),
    vad_options=SimpleNamespace(),
)

fake_segments = [
    make_fake_segment(" To jest test whispera."),
    make_fake_segment(" To jest drugie zdanie."),
    make_fake_segment(" To jest kolejne zdanie."),
]


def test_transcribe_returns_proper_artifact(default_whisper_config, mocker):
    mock_model = mocker.Mock()
    mock_model.transcribe.return_value = (iter(fake_segments), fake_info)

    result = transcribe(mock_model, default_whisper_config, "/tmp/path/rec_1.wav")
    final_text = "To jest test whispera. To jest drugie zdanie. To jest kolejne zdanie."

    assert result.model_dump(exclude={"transcribe_s"}) == TranscribeArtifact(
        rec_id="rec_1",
        final_text=final_text,
        words=[make_word(word) for word in final_text.split(" ")],
        transcribe_s=20.1,
    ).model_dump(exclude={"transcribe_s"})
    assert result.transcribe_s > 0


def test_validate_accepts_correct_proposal():
    text = "pracowałem w dżirze i githabie"
    prop = Replacement(
        r_from="dżirze", r_to="Jirze", base_term="Jira", position=13, source="slm"
    )

    accepted, rejected = validate_slm_replacements(text, [prop])

    assert accepted == [prop]
    assert rejected == []


def test_validate_badprefix_incorrect_proposal():
    text = "pracowałem w dżirze i githabie"
    prop = Replacement(
        r_from="dżirze", r_to="dżirze", base_term="Jira", position=13, source="slm"
    )

    accepted, rejected = validate_slm_replacements(text, [prop])

    assert accepted == []
    assert rejected == [prop]


def test_validate_badposition_incorrect_proposal():
    text = "pracowałem w dżirze i githabie"
    prop = Replacement(
        r_from="dżirze", r_to="Jirze", base_term="Jira", position=17, source="slm"
    )

    accepted, rejected = validate_slm_replacements(text, [prop])

    assert accepted == []
    assert rejected == [prop]


def test_validate_accepts_short_term():
    text = "wdrożyłem na ejdablju es"
    prop = Replacement(
        r_from="ejdablju es", r_to="AWS-ie", base_term="AWS", position=13, source="slm"
    )

    accepted, rejected = validate_slm_replacements(text, [prop])

    assert accepted == [prop]
    assert rejected == []


class FakeSlmClient:
    def __init__(self, response):
        self.response = response

    def chat(self, system, user, schema):
        return self.response


SLM_TEXT = "Potem robiłem API w grafkuelu."
SLM_CANDIDATE = Replacement(
    r_from="grafkuelu",
    r_to="GraphQL",
    base_term="GraphQL",
    position=20,
    source="dictionary",
)


def test_find_slm_returns_inflected_replacement(default_slm_config):
    client = FakeSlmClient('{"status": "podmiana", "r_to": "GraphQL-u"}')

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == [
        Replacement(
            r_from="grafkuelu",
            r_to="GraphQL-u",
            base_term="GraphQL",
            position=20,
            source="slm",
        )
    ]


def test_find_slm_skips_when_model_does_not_know(default_slm_config):
    client = FakeSlmClient('{"status": "nie_wiem", "r_to": ""}')

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == []


def test_find_slm_skips_truncated_json(default_slm_config):
    client = FakeSlmClient('{"status": "podm')

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == []


def test_find_slm_without_candidates_is_not_implemented(default_slm_config):
    client = FakeSlmClient('{"status": "podmiana", "r_to": "x"}')

    with pytest.raises(NotImplementedError):
        find_slm_replacements(SLM_TEXT, None, default_slm_config, client)


def test_find_slm_skips_json_without_keys(default_slm_config):
    client = FakeSlmClient("{}")

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == []


def test_find_slm_skips_replacement_without_r_to(default_slm_config):
    client = FakeSlmClient('{"status": "podmiana"}')

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == []


def test_find_slm_skips_unchanged_fragment(default_slm_config):
    client = FakeSlmClient('{"status": "podmiana", "r_to": "grafkuelu"}')

    result = find_slm_replacements(
        SLM_TEXT, [SLM_CANDIDATE], default_slm_config, client
    )

    assert result == []
