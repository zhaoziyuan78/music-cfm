import zipfile
from pathlib import Path

import pytest

from cfmusic.download.extraction import safe_extract_zip


def test_safe_zip_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("folder/value.txt", "ok")
    output = safe_extract_zip(archive, tmp_path / "output")
    assert (output / "folder/value.txt").read_text(encoding="utf-8") == "ok"


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_zip_extraction_can_strip_member_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("phrases/example.mid", "midi")
        bundle.writestr("__MACOSX/phrases/._example.mid", "metadata")

    output = safe_extract_zip(archive, tmp_path / "output", member_prefix="phrases")

    assert (output / "example.mid").read_text(encoding="utf-8") == "midi"
    assert not (output / "__MACOSX").exists()
