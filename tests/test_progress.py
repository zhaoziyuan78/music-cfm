from __future__ import annotations

from cfmusic.progress import progress_bar, progress_enabled, track


def test_progress_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CFMUSIC_PROGRESS", "0")
    monkeypatch.delenv("RANK", raising=False)
    assert not progress_enabled()
    assert list(track(range(3), description="test", total=3)) == [0, 1, 2]
    bar = progress_bar(description="test", total=2)
    assert bar.disable
    bar.update(2)
    bar.close()


def test_only_rank_zero_reports_progress(monkeypatch) -> None:
    monkeypatch.setenv("CFMUSIC_PROGRESS", "1")
    monkeypatch.setenv("RANK", "1")
    assert not progress_enabled()
    monkeypatch.setenv("RANK", "0")
    assert progress_enabled()
