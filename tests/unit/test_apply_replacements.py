import pytest

from log_runs import Replacement
from process import apply_replacements


def make_replacement(text, r_from, r_to):
    """Build a Replacement for the first occurrence of `r_from` in `text`."""
    position = text.index(r_from)
    return Replacement(
        r_from=r_from,
        r_to=r_to,
        base_term=r_to,
        position=position,
        source="dictionary",
    )


def test_returns_text_unchanged_without_replacements():
    text = "mamy taski w dżirze"

    results = apply_replacements(text, [])

    assert results == text


def test_changes_nothing_outside_replacement():
    text = "mamy taski w dżirze i obrazy"
    replacements = [make_replacement(text, "dżirze", "Jira")]

    results = apply_replacements(text, replacements)

    assert results == "mamy taski w Jira i obrazy"


def test_two_replacements_do_not_shift_positions():
    text = "mamy taski w dżirze i obrazy w dokerze"
    replacements = [
        make_replacement(text, "dokerze", "Docker"),
        make_replacement(text, "dżirze", "Jira"),
    ]

    results = apply_replacements(text, replacements)

    assert results == "mamy taski w Jira i obrazy w Docker"


def test_raises_on_overlapping_replacements():
    text = "robimy merge request"
    replacements = [
        make_replacement(text, "merge request", "merge request"),
        make_replacement(text, "merge", "merge"),
    ]

    with pytest.raises(ValueError):
        apply_replacements(text, replacements)


def test_touching_replacements_are_not_a_conflict():
    text = "dżiradoker"
    replacements = [
        make_replacement(text, "dżira", "Jira"),
        make_replacement(text, "doker", "Docker"),
    ]

    results = apply_replacements(text, replacements)

    assert results == "JiraDocker"


def test_raises_when_r_from_not_at_position():
    text = "mamy taski w dżirze"
    wrong = Replacement(
        r_from="dżirze",
        r_to="Jira",
        base_term="Jira",
        position=0,
        source="dictionary",
    )

    with pytest.raises(ValueError):
        apply_replacements(text, [wrong])
