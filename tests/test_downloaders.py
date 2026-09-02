from pathlib import Path

import pytest

from cfmusic.download.checksums import sha256_file, verify_sha256
from cfmusic.download.licenses import DatasetLicense, require_license_acknowledgement


def test_checksum_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"counterfactual music")
    digest = sha256_file(path)
    assert verify_sha256(path, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        verify_sha256(path, "0" * 64)


def test_unknown_license_has_two_gates() -> None:
    license_info = DatasetLicense("UNKNOWN_VERIFY_WITH_DATASET_AUTHORS", True)
    with pytest.raises(PermissionError, match=r"license\.accept"):
        require_license_acknowledgement(
            "xmidi", license_info, accept=False, acknowledge_unknown=False
        )
    with pytest.raises(PermissionError, match="acknowledge_unknown"):
        require_license_acknowledgement(
            "xmidi", license_info, accept=True, acknowledge_unknown=False
        )
    require_license_acknowledgement("xmidi", license_info, accept=True, acknowledge_unknown=True)


def test_per_file_checksum_progress_is_opt_in(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CFMUSIC_PROGRESS", "1")
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "small.mid"
    path.write_bytes(b"small MIDI payload")
    sha256_file(path)
    assert capsys.readouterr().err == ""
