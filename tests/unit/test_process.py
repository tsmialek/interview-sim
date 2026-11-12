from types import SimpleNamespace

from process import transcribe
from log_runs import TranscribeArtifact
from faster_whisper.transcribe import Word


def make_word(
    word: str,
    start: float = 0.0,
    end: float = 1.0,
    probability: float = 0.9
) -> Word:
    return Word(start, end, word, probability)


def make_fake_segment(
    text: str, start: float = 0.0, end: float = 1.0, words: list | None = None
):
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=words or [make_word(word) for word in text.strip().split(" ")]
    )


fake_info = SimpleNamespace(
    language="pl",
    language_probablity=0.95,
    duration=20.5,
    duration_after_vad=19,
    transcription_options=SimpleNamespace(),
    vad_options=SimpleNamespace())

fake_segments = [
    make_fake_segment(" To jest test whispera."),
    make_fake_segment(" To jest drugie zdanie."),
    make_fake_segment(" To jest kolejne zdanie."),
]


def test_transcribe_returns_proper_artifact(default_whisper_config, mocker):
    mock_model = mocker.Mock()
    mock_model.transcribe.return_value = (iter(fake_segments), fake_info)

    result = transcribe(
        mock_model, default_whisper_config, "/tmp/path/rec_1.wav")
    final_text = "To jest test whispera. To jest drugie zdanie. To jest kolejne zdanie."

    assert result == TranscribeArtifact(
        rec_id="rec_1",
        final_text=final_text,
        whisper_text=final_text,
        words=[make_word(word) for word in final_text.split(" ")]
    )
