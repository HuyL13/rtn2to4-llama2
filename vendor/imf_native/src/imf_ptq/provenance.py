import hashlib
import json
import subprocess
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

def assert_origin(path: Path, expected: str) -> str:
    try:
        actual = _git(path, "remote", "get-url", "origin").removesuffix(".git")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError(f"unexpected origin for {path}") from exc
    if actual != expected.removesuffix(".git"):
        raise ValueError(f"unexpected origin: {actual!r}, expected {expected!r}")
    return actual

def git_identity(path: Path, expected: str) -> dict[str, str]:
    return {"origin": assert_origin(path, expected), "commit": _git(path, "rev-parse", "HEAD")}

def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

