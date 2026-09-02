"""Durable scalar logs, TensorBoard events, and live training curves."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    """Persist scalar metrics and periodically replace a compact curve dashboard."""

    def __init__(
        self,
        run_dir: Path,
        *,
        append: bool = True,
        curve_interval: int = 50,
    ) -> None:
        if curve_interval <= 0:
            raise ValueError("curve_interval must be positive")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.jsonl = run_dir / "metrics.jsonl"
        self.csv_path = run_dir / "metrics.csv"
        self.text_path = run_dir / "training.log"
        self.curve_path = run_dir / "training_curves.png"
        if not append:
            self.curve_path.unlink(missing_ok=True)
        self.curve_interval = curve_interval
        self._history: list[dict[str, float | int]] = []
        if append and self.jsonl.exists():
            for line in self.jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    restored = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(restored, dict):
                    numeric = {
                        str(key): value
                        for key, value in restored.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    }
                    self._history.append(numeric)
        csv_exists = append and self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._fieldnames: list[str] | None = None
        if csv_exists:
            with self.csv_path.open(newline="", encoding="utf-8") as stream:
                self._fieldnames = next(csv.reader(stream), None)
        mode = "a" if append else "w"
        self._json_stream = self.jsonl.open(mode, encoding="utf-8")
        self._csv_stream = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._text_stream = self.text_path.open(mode, encoding="utf-8")
        self._csv_writer: csv.DictWriter[str] | None = None
        self._csv_exists = csv_exists
        tensorboard_dir = run_dir / "tensorboard"
        if not append and tensorboard_dir.exists():
            shutil.rmtree(tensorboard_dir)
        self._tensorboard = SummaryWriter(log_dir=str(tensorboard_dir))

    def log(self, metrics: Mapping[str, float | int]) -> None:
        row = dict(metrics)
        self._json_stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._text_stream.write(" ".join(f"{key}={value}" for key, value in row.items()) + "\n")
        if self._csv_writer is None:
            if self._fieldnames is not None:
                # A resumed run may add timing fields. Keep the historical CSV valid;
                # JSONL retains every newly introduced scalar without a fixed schema.
                fieldnames = self._fieldnames
            else:
                fieldnames = list(row)
                self._fieldnames = fieldnames
            self._csv_writer = csv.DictWriter(
                self._csv_stream, fieldnames=fieldnames, extrasaction="ignore"
            )
            if not self._csv_exists:
                self._csv_writer.writeheader()
                self._csv_exists = True
        self._csv_writer.writerow(row)
        self._history.append(row)
        step = int(row.get("step", len(self._history)))
        for key, value in row.items():
            if key != "step":
                self._tensorboard.add_scalar(key, value, step)
        self.flush()
        if len(self._history) % self.curve_interval == 0:
            self.render_curves()

    def flush(self) -> None:
        self._json_stream.flush()
        self._csv_stream.flush()
        self._text_stream.flush()

    def render_curves(self) -> None:
        """Atomically replace a PNG dashboard derived from all recorded rows."""

        if not self._history:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        excluded = {"step", "epoch", "sequence_length", "abduction_steps"}
        keys = sorted({key for row in self._history for key in row if key not in excluded})
        if not keys:
            return
        columns = min(3, len(keys))
        rows = math.ceil(len(keys) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.2 * rows), squeeze=False)
        for axis, key in zip(axes.flat, keys, strict=False):
            points = [
                (int(row.get("step", index + 1)), float(row[key]))
                for index, row in enumerate(self._history)
                if key in row and math.isfinite(float(row[key]))
            ]
            if not points:
                axis.set_visible(False)
                continue
            steps = np.asarray([point[0] for point in points])
            values = np.asarray([point[1] for point in points])
            axis.plot(steps, values, linewidth=0.8, alpha=0.35, color="tab:blue")
            window = min(50, max(1, len(values) // 20))
            if window > 1:
                kernel = np.ones(window) / window
                smoothed = np.convolve(values, kernel, mode="valid")
                axis.plot(steps[window - 1 :], smoothed, linewidth=1.5, color="tab:blue")
            axis.set_title(key)
            axis.set_xlabel("step")
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(keys) :]:
            axis.set_visible(False)
        figure.suptitle(self.run_dir.name)
        figure.tight_layout()
        temporary = self.curve_path.with_suffix(".png.tmp")
        figure.savefig(temporary, format="png", dpi=120)
        plt.close(figure)
        temporary.replace(self.curve_path)

    def close(self) -> None:
        if not self._json_stream.closed:
            self.render_curves()
            self.flush()
            self._tensorboard.flush()
            self._tensorboard.close()
            self._json_stream.close()
            self._csv_stream.close()
            self._text_stream.close()
