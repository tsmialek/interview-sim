import pytest

from utils import get_sha256

TEST_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


def test_get_sha256_returns_hash():
    assert get_sha256("test") == TEST_SHA256


@pytest.mark.parametrize(
    "length, expected",
    [
        (8, "9f86d081"),
        (16, "9f86d081884c7d65"),
        (64, TEST_SHA256),
        (100, TEST_SHA256),
    ],
)
def test_get_sha256_return_hash_custom_lengths(length, expected):
    assert get_sha256("test", length) == expected
