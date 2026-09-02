from pathlib import Path

from cfmusic.logging import MetricLogger


def test_metric_logger_resets_fresh_runs_and_appends_resumes(tmp_path: Path) -> None:
    first = MetricLogger(tmp_path, append=False)
    first.log({"step": 1, "loss": 2.0})
    first.close()
    assert (tmp_path / "training.log").is_file()
    assert (tmp_path / "training_curves.png").is_file()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))

    resumed = MetricLogger(tmp_path, append=True)
    resumed.log({"step": 2, "loss": 1.0})
    resumed.close()
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    restarted = MetricLogger(tmp_path, append=False)
    restarted.log({"step": 1, "loss": 3.0})
    restarted.close()
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"loss": 3.0' in lines[0]
    assert len((tmp_path / "training.log").read_text(encoding="utf-8").splitlines()) == 1
