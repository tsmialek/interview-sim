from validate_data import main

TERMS_CSV = (
    "term,aliases,category,description\n"
    "Jenkins,,tool,CI server\n"
    "Redis,,technology,cache\n"
)
RECORDINGS_CSV = "file_id,speaker,sample_type,question_id,style,device,environment,split\n"


def make_data_dir(tmp_path, hidden_terms=None):
    """Write a minimal valid dataset; hidden_terms=None leaves hidden_terms.txt out."""
    (tmp_path / "terms_it.csv").write_text(TERMS_CSV, encoding="utf-8")
    (tmp_path / "recordings.csv").write_text(RECORDINGS_CSV, encoding="utf-8")
    if hidden_terms is not None:
        (tmp_path / "hidden_terms.txt").write_text(hidden_terms, encoding="utf-8")
    return tmp_path


def run_main(data_dir):
    return main(["--data-dir", str(data_dir), "--skip-audio"])


def test_passes_for_valid_hidden_terms(tmp_path):
    data_dir = make_data_dir(tmp_path, "Jenkins\n\nredis\n")

    results = run_main(data_dir)

    assert results == 0


def test_reports_hidden_term_with_typo(tmp_path, capsys):
    data_dir = make_data_dir(tmp_path, "Jenkis\n")

    results = run_main(data_dir)

    assert results == 1
    assert "'Jenkis' is not a term in terms_it.csv, did you mean 'Jenkins'?" in capsys.readouterr().out


def test_reports_missing_hidden_terms_file(tmp_path, capsys):
    data_dir = make_data_dir(tmp_path)

    results = run_main(data_dir)

    assert results == 1
    output = capsys.readouterr().out
    assert "hidden_terms.txt" in output
    assert "file not found" in output


def test_reports_old_comma_separated_format(tmp_path, capsys):
    data_dir = make_data_dir(tmp_path, "Jenkins,Redis\n")

    results = run_main(data_dir)

    assert results == 1
    assert "'Jenkins,Redis' is not a term" in capsys.readouterr().out
