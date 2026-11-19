import pytest

from project_data import ProjectRepository


def test_get_recordings_returns_recordings_for_split_from_csv(make_data_dir):
    data_dir = make_data_dir(["rec_201", "rec_202", "rec_203", "rec_204"])
    repo = ProjectRepository(data_dir)
    result = repo.get_recordings("dev")

    assert result == [
        (data_dir / "audio" / "rec_201.wav"),
        (data_dir / "audio" / "rec_203.wav"),
    ]


def test_get_recordings_raises_value_error_if_rec_from_csv_not_in_data_folder(
    make_data_dir,
):
    data_dir = make_data_dir(["rec_201", "rec_202", "rec_204"])
    repo = ProjectRepository(data_dir)
    with pytest.raises(ValueError, match="rec_203"):
        repo.get_recordings("dev")


def test_get_reference_returns_proper_text(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)
    result = repo.get_reference("rec_1")
    assert result == "Korzystałem z {AWS|AWS} z systemów {EC2|EC2}"


def test_get_reference_raises_value_error_if_reference_transcript_not_found(
    make_data_dir,
):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)
    with pytest.raises(ValueError, match="rec_2"):
        repo.get_reference("rec_2")


def test_get_hidden_terms(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)
    result = repo.get_hidden_terms()
    assert result == ["Jira", "GitHub"]


def test_get_hidden_terms_raises_exception_if_file_is_missing(tmp_path):
    repo = ProjectRepository(tmp_path)
    with pytest.raises(FileNotFoundError, match="hidden_terms"):
        repo.get_hidden_terms()


def test_get_dictionary_returns_proper_DataFrame_exclude_hidden_false(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)
    terms_dict = repo.get_dictionary(exclude_hidden=False)
    row = terms_dict.loc[terms_dict["term"] == "Jira"].iloc[0].to_dict()

    assert row == {
        "term": "Jira",
        "aliases": ["dżira", "dżiry", "dżirze", "dżirą", "dżirę", "jiry", "jirze"],
        "category": "tool",
        "description": "system do zarządzania zadaniami",
    }


def test_get_dictionary_returns_proper_DataFrame_exclude_hidden_true(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)
    terms_dict = repo.get_dictionary()
    proper_row = terms_dict.loc[terms_dict["term"] == "Slack"].iloc[0].to_dict()

    assert proper_row == {
        "term": "Slack",
        "aliases": ["slak", "slaka", "slaku", "slacku", "slackiem", "slakiem"],
        "category": "tool",
        "description": "komunikator zespołowy",
    }
    assert len(terms_dict[terms_dict["term"].isin(["Jira", "GitHub"])]) == 0


def test_get_dictionary_raises_exception_if_aliases_are_empty(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)

    with open(data_dir / "terms_it.csv", "a", encoding="utf-8") as file:
        file.write("PostgreSQL,,technology,relacyjna baza danych")

    with pytest.raises(ValueError, match="PostgreSQL"):
        repo.get_dictionary()


def test_get_dictionary_raises_exception_if_terms_are_empty(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)

    with open(data_dir / "terms_it.csv", "a", encoding="utf-8") as file:
        file.write(",test|test|test,technology,relacyjna baza danych")

    with pytest.raises(ValueError, match="8"):
        repo.get_dictionary()


def test_get_dictionary_sha(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)

    first_sha = repo.get_dictionary_sha()
    second_sha_no_change = repo.get_dictionary_sha()

    with open(data_dir / "terms_it.csv", "a", encoding="utf-8") as file:
        file.write("PostgreSQL,,technology,relacyjna baza danych")
    sha_after_change = repo.get_dictionary_sha()

    with open(data_dir / "hidden_terms.txt", "a", encoding="utf-8") as file:
        file.write("Test")
    sha_after_another_file_change = repo.get_dictionary_sha()

    assert first_sha == second_sha_no_change
    assert first_sha != sha_after_change
    assert sha_after_change == sha_after_another_file_change


def test_get_hidden_terms_sha(make_data_dir):
    data_dir = make_data_dir()
    repo = ProjectRepository(data_dir)

    first_sha = repo.get_hidden_terms_sha()
    second_sha_no_change = repo.get_hidden_terms_sha()

    with open(data_dir / "hidden_terms.txt", "a", encoding="utf-8") as file:
        file.write("Test")
    sha_after_change = repo.get_hidden_terms_sha()

    with open(data_dir / "terms_it.csv", "a", encoding="utf-8") as file:
        file.write("PostgreSQL,,technology,relacyjna baza danych")
    sha_after_another_file_change = repo.get_hidden_terms_sha()

    assert first_sha == second_sha_no_change
    assert first_sha != sha_after_change
    assert sha_after_change == sha_after_another_file_change
