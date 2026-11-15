from pathlib import Path
import subprocess
import hashlib
from importlib.metadata import distributions


def get_last_commit(repo_path: Path | str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def get_git_changed_files(repo_path: Path | str) -> str:
    result = subprocess.run(
        ["git", "status", "-u", "-z", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    files = [f for f in result.stdout.split("\0") if f]
    return "\n".join([file for file in files]).strip()


def get_current_diffs(repo_path: Path | str) -> str:
    changed_files = get_git_changed_files(repo_path)
    if not changed_files:
        return ""

    diffs = []

    for line in changed_files.splitlines():
        if not line.strip():
            continue

        mode = line[:3]
        file_path = line[3:].strip()

        cmd = ["git", "--no-pager", "diff"]
        if mode in ["?? "]:
            cmd.extend(["--no-index", "/dev/null", "--", f"{file_path}"])
        else:
            cmd.extend(["HEAD", "--", f"{file_path}"])

        res = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        if res.returncode in (0, 1):
            if res.stdout.strip():
                diffs.append(res.stdout.strip())
        else:
            raise RuntimeError(f"Error when checking changed files: {res.stderr}")

    return "\n\n".join(diffs)


def get_project_environment(
    packages: list[str] | None = None,
) -> dict[str, str]:
    target_packages = (
        {pkg.lower().replace("_", "-") for pkg in packages} if packages else None
    )

    result: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue

        normalized_name = name.lower().replace("_", "-")
        if target_packages is None or normalized_name in target_packages:
            result[name] = dist.version

    return result


def get_sha256(text: str, length: int | None = None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest
