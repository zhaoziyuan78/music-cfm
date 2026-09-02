"""Resumable HTTP archive downloader."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import requests
from filelock import FileLock

from cfmusic.download.base import DownloadError, DownloadResult
from cfmusic.download.checksums import verify_sha256
from cfmusic.download.extraction import safe_extract_zip
from cfmusic.progress import progress_bar


class HttpArchiveSource:
    def __init__(
        self,
        dataset: str,
        url: str,
        archive_name: str,
        expected_sha256: str | None,
        *,
        chunk_size: int = 1024 * 1024,
        timeout: int = 60,
    ) -> None:
        self.dataset = dataset
        self.url = url
        self.archive_name = archive_name
        self.expected_sha256 = expected_sha256
        self.chunk_size = chunk_size
        self.timeout = timeout

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
        if dry_run:
            return self._result(archive, extract_root, None, 0)
        with FileLock(str(destination / ".download.lock")):
            if force:
                archive.unlink(missing_ok=True)
                (destination / f"{self.archive_name}.part").unlink(missing_ok=True)
            if archive.exists():
                digest = verify_sha256(archive, self.expected_sha256)
            else:
                self._transfer(archive, resume=resume)
                digest = verify_sha256(archive, self.expected_sha256)
            safe_extract_zip(archive, extract_root, force=force)
        return self._result(archive, extract_root, digest, archive.stat().st_size)

    def _transfer(self, archive: Path, *, resume: bool) -> None:
        partial = archive.with_suffix(archive.suffix + ".part")
        offset = partial.stat().st_size if resume and partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                self.url, headers=headers, stream=True, timeout=self.timeout, allow_redirects=True
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != requests.codes.partial_content:
                    offset = 0
                content_length = int(response.headers.get("content-length", "0"))
                required = content_length + offset
                free = shutil.disk_usage(archive.parent).free
                if required and free < required * 1.05:
                    raise DownloadError(
                        f"Insufficient disk space: need about {required} bytes, have {free}"
                    )
                mode = "ab" if offset else "wb"
                with (
                    partial.open(mode) as stream,
                    progress_bar(
                        description=f"Download {archive.name}",
                        total=required or None,
                        initial=offset,
                        unit="B",
                        position=1,
                    ) as progress,
                ):
                    progress.unit_scale = True
                    progress.unit_divisor = 1024
                    for chunk in response.iter_content(self.chunk_size):
                        if chunk:
                            stream.write(chunk)
                            progress.update(len(chunk))
            partial.replace(archive)
        except requests.RequestException as error:
            raise DownloadError(f"HTTP download failed for {self.url}: {error}") from error

    def _result(
        self, archive: Path, extract_root: Path, digest: str | None, size: int
    ) -> DownloadResult:
        return DownloadResult(
            dataset=self.dataset,
            source=self.url,
            resolved_revision=None,
            archive_path=str(archive),
            size_bytes=size,
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
