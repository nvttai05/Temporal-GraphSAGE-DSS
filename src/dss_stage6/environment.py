from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = [
    "joblib",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "psutil",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tqdm",
    "xgboost",
]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_environment() -> dict[str, object]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": _package_versions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_environment(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

