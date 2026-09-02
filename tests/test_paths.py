from pathlib import Path

from cfmusic.paths import resolve_path


def test_resolve_path_is_absolute(tmp_path: Path, monkeypatch: object) -> None:
    assert resolve_path(tmp_path).is_absolute()
    assert resolve_path("relative/place").is_absolute()
