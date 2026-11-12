from pathlib import Path
from typing import Dict, Literal
import subprocess
import hashlib
from importlib.metadata import distributions


def get_last_commit(repo_path: Path | str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout.strip()


def get_git_changed_files(repo_path: Path | str) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout.strip()


def get_current_diffs(repo_path: Path | str) -> str:
    changed_files = get_git_changed_files(repo_path)
    if not changed_files:
        return ""

    diffs = []

    for line in changed_files.splitlines():
        if not line.strip():
            continue

        file_path = line.strip().split()[-1]
        res = subprocess.run(
            ["git", "--no-pager", "diff", "--no-index", "/dev/null", file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        if res.returncode in (0, 1):
            if res.stdout.strip():
                diffs.append(res.stdout.strip())
        else:
            diffs.append(res.stderr.strip())

    return "\n\n".join(diffs)


def get_project_environment() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version
        for dist in distributions()
        if dist.metadata.get("Name")
    }


def append_jsonl(file_path: Path | str, row: str) -> None:
    pass


def append_csv(file_path: Path | str, row: str) -> None:
    pass


def get_sha256(text: str, length: int | None = None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest
