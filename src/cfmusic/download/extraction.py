"""Archive extraction guarded against zip-slip and partial writes."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from cfmusic.progress import track


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    force: bool = False,
    member_prefix: str | None = None,
) -> Path:
    """Extract a ZIP atomically, optionally stripping one member path prefix."""

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            prefix_parts = PurePosixPath(member_prefix).parts if member_prefix else ()
            members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for member in bundle.infolist():
                member_path = PurePosixPath(member.filename)
                if prefix_parts and member_path.parts[: len(prefix_parts)] != prefix_parts:
                    continue
                relative_parts = member_path.parts[len(prefix_parts) :]
                if not relative_parts:
                    continue
                members.append((member, PurePosixPath(*relative_parts)))
            for member, relative_path in track(
                members,
                description=f"Extract {archive.name}",
                total=len(members),
                unit="file",
                position=1,
            ):
                target = (temp_root / Path(*relative_path.parts)).resolve()
                if not target.is_relative_to(temp_root.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
        if destination.exists():
            if force:
                shutil.rmtree(destination)
            else:
                return destination
        temp_root.replace(destination)
        return destination
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
