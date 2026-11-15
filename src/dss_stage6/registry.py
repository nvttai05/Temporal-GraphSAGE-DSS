from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunRegistry:
    """Write immutable run receipts with config, code, environment and artifacts."""

    def __init__(self, root: str | Path, registry_dir: str | Path):
        self.root = Path(root).resolve()
        self.registry_dir = Path(registry_dir).resolve()
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def code_version(self) -> dict[str, str]:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE_NOT_A_GIT_WORKTREE"
        source_hashes = {
            str(path.relative_to(self.root)): _sha256(path)
            for path in sorted((self.root / "src" / "dss_stage6").rglob("*.py"))
        }
        return {"git_commit": commit, "source_tree_sha256": _canonical_hash(source_hashes)}

    def register(
        self,
        *,
        run_id: str,
        config: dict[str, Any],
        seeds: dict[str, Any],
        started_monotonic: float,
        artifacts: Iterable[str | Path],
        status: str,
    ) -> Path:
        artifact_hashes: dict[str, str] = {}
        for artifact in artifacts:
            path = Path(artifact).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact_hashes[str(path.relative_to(self.root))] = _sha256(path)
        receipt = {
            "run_id": run_id,
            "status": status,
            "config": config,
            "config_sha256": _canonical_hash(config),
            "code_version": self.code_version(),
            "seeds": seeds,
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "runtime_seconds": time.monotonic() - started_monotonic,
            "artifact_sha256": artifact_hashes,
        }
        destination = self.registry_dir / f"{run_id}.json"
        if destination.exists():
            raise FileExistsError(f"Run id already registered: {run_id}")
        destination.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return destination
