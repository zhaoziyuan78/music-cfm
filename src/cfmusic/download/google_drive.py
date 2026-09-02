"""Google Drive source built on gdown with atomic completion."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from filelock import FileLock
from gdown import download as gdown_download  # type: ignore[attr-defined]

from cfmusic.download.base import DownloadError, DownloadResult
from cfmusic.download.checksums import verify_sha256
from cfmusic.download.extraction import safe_extract_zip


class GoogleDriveSource:
    def __init__(
        self, dataset: str, file_id: str, archive_name: str, expected_sha256: str | None
    ) -> None:
        self.dataset = dataset
        self.file_id = file_id
        self.archive_name = archive_name
        self.expected_sha256 = expected_sha256

    def download(
        self,
        destination: Path,
        *,
        resume: bool,
        force: bool,
        dry_run: bool,
    ) -> DownloadResult:
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / self.archive_name
        extract_root = destination / "extracted"
        digest: str | None = None
        if not dry_run:
            with FileLock(str(destination / ".download.lock")):
                if force:
                    archive.unlink(missing_ok=True)
                if not archive.exists():
                    partial = archive.with_suffix(archive.suffix + ".part")
                    result = gdown_download(
                        id=self.file_id, output=str(partial), resume=resume, quiet=False
                    )
                    if result is None:
                        raise DownloadError(
                            "Google Drive download failed (possibly quota-limited). Download file ID "
                            f"{self.file_id} manually as {archive}, then rerun the command."
                        )
                    partial.replace(archive)
                digest = verify_sha256(archive, self.expected_sha256)
                safe_extract_zip(archive, extract_root, force=force)
        return DownloadResult(
            dataset=self.dataset,
            source=f"gdrive:{self.file_id}",
            resolved_revision=None,
            archive_path=str(archive),
            size_bytes=archive.stat().st_size if archive.exists() else 0,
            sha256=digest,
            downloaded_at=dt.datetime.now(dt.UTC).isoformat(),
            license_acknowledged=True,
            extract_root=str(extract_root),
            checksum_source=(
                "official" if self.expected_sha256 else "locally_computed_not_official"
            )
            if digest
            else None,
        )
