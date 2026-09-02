"""Dataset download source contracts and common result metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    source: str
    resolved_revision: str | None
    archive_path: str | None
    size_bytes: int
    sha256: str | None
    downloaded_at: str
    license_acknowledged: bool
    extract_root: str
    checksum_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DatasetSource(Protocol):
    def download(
        self,
        destination: Path,
        *,
        resume: bool,
        force: bool,
        dry_run: bool,
    ) -> DownloadResult:
        """Download and extract a dataset into destination."""


class DownloadError(RuntimeError):
    """Raised for actionable download failures."""
