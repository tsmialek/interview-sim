from pathlib import Path
from typing import Dict, Literal
import subprocess
import hashlib


def get_git_info(cmd: Literal["commit", "files_changed", "diff"], repo_path: Path | str) -> str:
    """Get git info
    Args:
        cmd (str): commit, files_changed, diff
        repo_path (str): repository root folder path
    """
    match cmd:
        case "commit":
            cli_cmd = ["git", "rev-parse", "HEAD"]
        case "files_changed":
            cli_cmd = ["git", "status", "--porcelain"]
        case "diff":
            cli_cmd = ["git", "--no-pager", "diff", "HEAD"]
        case _:
            raise ValueError(
                f"Wrond command: {cmd}, You can use only 'commit', 'files_changed', 'diff'")

    return subprocess.run(
        cli_cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()


def get_project_environment() -> Dict[str, str]:
    pass


def get_sha256(text: str, length: int | None = None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest
