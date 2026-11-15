from utils import get_last_commit, get_git_changed_files, get_current_diffs
import subprocess


def test_get_last_commit_proper_commit(git_repo):
    commit_hash = get_last_commit(git_repo)

    print(commit_hash)
    assert len(commit_hash) == 40


def test_get_git_changed_files_returns_all_files(git_repo):
    new_dir = git_repo / "new_dir"
    new_dir.mkdir()

    new_file = git_repo / "new_file.json"
    new_file.write_text("New file test")
    subprocess.run(["git", "add", str(new_file)], cwd=git_repo, check=True)

    (new_dir / "file in dir.txt").write_text("test for file in dir")

    (git_repo / "test_file_0.txt").unlink()

    expected = ["A  new_file.json", "?? new_dir/file in dir.txt", " D test_file_0.txt"]
    result = get_git_changed_files(git_repo).splitlines()

    print(result)
    assert all(ex in result for ex in expected)


def test_get_git_current_diffs_returns_all_diffs(git_repo):
    new_dir = git_repo / "new_dir"
    new_dir.mkdir()

    new_file = git_repo / "new_file.json"
    new_file.write_text("New file test\n")
    subprocess.run(["git", "add", str(new_file)], cwd=git_repo, check=True)

    (new_dir / "file in dir.txt").write_text("test for file in dir\n")

    with open(git_repo / "test_file_1.txt", "a+") as file:
        file.write("Changed line\n")

    (git_repo / "test_file_0.txt").unlink()

    diffs = get_current_diffs(git_repo)
    print(get_git_changed_files(git_repo))
    print(diffs)

    assert diffs.count("-Hello world") == 1
    assert diffs.count("New file test") == 1
    assert diffs.count("+Hello world from second file") == 0
    assert diffs.count("+Changed line") == 1
    assert diffs.count("+test for file in dir") == 1


import subprocess
from pathlib import Path

from utils import get_current_diffs


def snapshot(root: Path) -> dict[str, bytes]:
    """Returns {relative path: content} for every file except .git."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def test_patch_recreates_working_tree(git_repo, tmp_path):
    staged = git_repo / "new_file.json"
    staged.write_text("New file test\n", encoding="utf-8")
    subprocess.run(["git", "add", str(staged)], cwd=git_repo, check=True)

    (git_repo / "new_dir").mkdir()
    (git_repo / "new_dir" / "file in dir.txt").write_text("in dir\n", encoding="utf-8")

    with open(git_repo / "test_file_1.txt", "a", encoding="utf-8") as f:
        f.write("Changed line\n")

    (git_repo / "test_file_0.txt").unlink()

    patch_file = tmp_path / "code.patch"
    patch_file.write_text(get_current_diffs(git_repo) + "\n", encoding="utf-8")

    clean = tmp_path / "clean"
    subprocess.run(["git", "clone", "-q", str(git_repo), str(clean)], check=True)
    subprocess.run(["git", "apply", str(patch_file)], cwd=clean, check=True)

    assert snapshot(clean) == snapshot(git_repo)
