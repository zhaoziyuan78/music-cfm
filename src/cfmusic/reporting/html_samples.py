"""Static blinded human-evaluation HTML and answer-sheet export."""

from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path


def export_human_evaluation(
    comparisons: list[dict[str, str]], output_dir: Path, *, seed: int = 2026
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    randomized: list[dict[str, str]] = []
    blocks: list[str] = []
    for index, comparison in enumerate(comparisons):
        methods = [("A", comparison["method_a"]), ("B", comparison["method_b"])]
        rng.shuffle(methods)
        row = {
            "question_id": str(index),
            "source": comparison["source"],
            "target_style": comparison["target_style"],
            "shown_a": methods[0][1],
            "shown_b": methods[1][1],
            "identity_a": methods[0][0],
            "identity_b": methods[1][0],
        }
        randomized.append(row)
        blocks.append(
            f"<section><h2>Question {index + 1}: target {html.escape(row['target_style'])}</h2>"
            f"<p>Source: <a href='{html.escape(row['source'])}'>MIDI</a></p>"
            f"<p>Method A: <a href='{html.escape(row['shown_a'])}'>MIDI</a></p>"
            f"<p>Method B: <a href='{html.escape(row['shown_b'])}'>MIDI</a></p></section>"
        )
    document = "<!doctype html><meta charset='utf-8'><title>CFMusic evaluation</title>" + "\n".join(
        blocks
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    (output_dir / "sample_manifest.json").write_text(
        json.dumps({"seed": seed, "questions": randomized}, indent=2), encoding="utf-8"
    )
    with (output_dir / "answer_sheet.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["question_id", "preferred", "style_a", "style_b", "comments"]
        )
        writer.writeheader()
        for row in randomized:
            writer.writerow({"question_id": row["question_id"]})
