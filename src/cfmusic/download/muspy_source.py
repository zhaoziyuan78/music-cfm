"""MusPy-backed EMOPIA source."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from filelock import FileLock

from cfmusic.download.base import DownloadResult


class MuspySource:
    def __init__(self, dataset: str, dataset_class: str) -> None:
        self.dataset = dataset
        self.dataset_class = dataset_class

    def download(
        self,
        destination: Path,
        *,
        resume: bool,
        force: bool,
        dry_run: bool,
    ) -> DownloadResult:
        destination.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            import muspy

            dataset_type = getattr(muspy, self.dataset_class)
            with FileLock(str(destination / ".download.lock")):
                dataset_type(root=str(destination), download_and_extract=True)
        return DownloadResult(
            dataset=self.dataset,
            source=f"muspy.{self.dataset_class}",
            resolved_revision=None,
            archive_path=None,
            size_bytes=0,
            sha256=None,
            downloaded_at=dt.datetime.now(dt.UTC).isoformat(),
            license_acknowledged=True,
            extract_root=str(destination),
        )
