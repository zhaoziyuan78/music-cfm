"""Shallow git dataset source."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

from filelock import FileLock

from cfmusic.download.base import DownloadError, DownloadResult


class GitSource:
    def __init__(self, dataset: str, repository: str, revision: str = "master") -> None:
        self.dataset = dataset
        self.repository = repository
        self.revision = revision

    def download(
        self,
        destination: Path,
        *,
        resume: bool,
        force: bool,
        dry_run: bool,
    ) -> DownloadResult:
        clone_dir = destination / "repository"
        destination.mkdir(parents=True, exist_ok=True)
        revision: str | None = None
        if not dry_run:
            with FileLock(str(destination / ".download.lock")):
                if force and clone_dir.exists():
                    raise DownloadError(
                        f"Refusing to delete existing git directory {clone_dir}; move it aside first"
                    )
                if not clone_dir.exists():
                    command = [
                        "git",
                        "clone",
                        "--progress",
                        "--depth",
                        "1",
                        "--branch",
                        self.revision,
                        self.repository,
                        str(clone_dir),
                    ]
                    clone_result = subprocess.run(command, check=False)
                    if clone_result.returncode != 0:
                        raise DownloadError(
                            f"git clone failed with exit code {clone_result.returncode}; "
                            "see Git output above"
                        )
                revision_result = subprocess.run(
                    ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if revision_result.returncode != 0:
                    raise DownloadError(
                        f"Cannot resolve git commit: {revision_result.stderr.strip()}"
                    )
                revision = revision_result.stdout.strip()
        return DownloadResult(
            dataset=self.dataset,
            source=self.repository,
            resolved_revision=revision,
            archive_path=None,
            size_bytes=0,
            sha256=None,
            downloaded_at=dt.datetime.now(dt.UTC).isoformat(),
            license_acknowledged=True,
            extract_root=str(clone_dir),
        )
