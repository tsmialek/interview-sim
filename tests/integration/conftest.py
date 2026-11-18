import pytest
import subprocess
from pathlib import Path


@pytest.fixture
def git_repo(tmp_path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "user@example.com"], cwd=repo_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True
    )

    dummy_file_0 = repo_dir / "test_file_0.txt"
    dummy_file_1 = repo_dir / "test_file_1.txt"
    dummy_file_0.write_text("Hello world!\n", encoding="utf-8")
    dummy_file_1.write_text("Hello world from second file\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Iinitial commit"], cwd=repo_dir, check=True)

    return repo_dir
