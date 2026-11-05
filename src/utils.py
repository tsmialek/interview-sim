from typing import Dict
import hashlib


def get_last_commit_hash() -> str:
    pass


def is_worktree_dirty() -> bool:
    pass


def get_current_git_dif() -> str:
    pass


def get_project_environment() -> Dict[str, str]:
    pass


def get_sha256(text: str, length: int | None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest
