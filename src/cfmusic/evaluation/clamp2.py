"""CLaMP 2 MIDI/text embedding evaluation through the official extractor."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mido
import numpy as np

from cfmusic.progress import track

CLAMP2_WEIGHT_FILENAMES = (
    "weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_"
    "t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth",
    "weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_"
    "lr_0.0001_batch_16_mask_0.45.pth",
)


def style_prompt(label: str, template: str = "This is a piece of {style} music.") -> str:
    if "{style}" not in template:
        raise ValueError("CLaMP 2 style template must contain {style}")
    return template.format(style=label.replace("_", " "))


def midi_to_mtf(source: Path, destination: Path) -> None:
    """Convert MIDI to the lossless MTF representation expected by CLaMP 2/M3."""

    midi = mido.MidiFile(source)
    lines = [f"ticks_per_beat {midi.ticks_per_beat}"]
    ignored_meta = {
        "text",
        "copyright",
        "track_name",
        "instrument_name",
        "lyrics",
        "marker",
        "cue_marker",
        "device_name",
        "sequencer_specific",
    }
    for message in midi.merged_track:
        if (message.is_meta and message.type in ignored_meta) or message.type == "sysex":
            continue
        encoded = " ".join(str(value) for value in message.dict().values())
        lines.append(encoded.encode("unicode_escape").decode("utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _normalized_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"CLaMP 2 produced an invalid embedding: {path}")
    return feature / norm


def extract_clamp2_embeddings(
    *,
    repository: Path,
    midi_files: Mapping[str, Path],
    texts: Mapping[str, str],
    python_executable: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run the archived official CLaMP 2 extractor once for a complete evaluation."""

    extractor = repository / "code" / "extract_clamp2.py"
    if not extractor.is_file():
        raise FileNotFoundError(
            f"CLaMP 2 extractor not found at {extractor}. Clone "
            "https://github.com/sanderwood/clamp2 and place its released weights in code/."
        )
    missing_weights = [
        repository / "code" / filename
        for filename in CLAMP2_WEIGHT_FILENAMES
        if not (repository / "code" / filename).is_file()
    ]
    if missing_weights:
        raise FileNotFoundError(
            "CLaMP 2 is missing its released checkpoint(s): "
            + ", ".join(str(path) for path in missing_weights)
        )
    with tempfile.TemporaryDirectory(prefix="cfmusic-clamp2-") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        outputs = root / "outputs"
        inputs.mkdir()
        for key, source in track(
            midi_files.items(),
            description="Convert MIDI to CLaMP 2 MTF",
            total=len(midi_files),
            unit="midi",
        ):
            midi_to_mtf(source, inputs / f"{key}.mtf")
        for key, value in texts.items():
            (inputs / f"{key}.txt").write_text(value, encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(repository / "code"), environment.get("PYTHONPATH", "")])
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(cache_dir)
        subprocess.run(
            [
                python_executable or sys.executable,
                str(extractor),
                str(inputs),
                str(outputs),
                "--normalize",
            ],
            cwd=repository / "code",
            env=environment,
            check=True,
        )
        expected = [*midi_files, *texts]
        missing = [key for key in expected if not (outputs / f"{key}.npy").is_file()]
        if missing:
            raise RuntimeError(f"CLaMP 2 did not produce embeddings for: {missing[:10]}")
        return {key: _normalized_feature(outputs / f"{key}.npy") for key in expected}


def clamp2_style_metrics(
    music_embedding: np.ndarray,
    *,
    style_embeddings: Sequence[np.ndarray],
    source_style_id: int,
    target_style_id: int,
) -> dict[str, float]:
    similarities = np.asarray(
        [float(music_embedding @ style_embedding) for style_embedding in style_embeddings]
    )
    if target_style_id >= len(similarities) or source_style_id >= len(similarities):
        raise ValueError("Artifact style id is outside the CLaMP 2 text vocabulary")
    return {
        "clamp2_target_similarity": float(similarities[target_style_id]),
        "clamp2_source_similarity": float(similarities[source_style_id]),
        "clamp2_target_minus_source": float(
            similarities[target_style_id] - similarities[source_style_id]
        ),
        "clamp2_target_style_success": float(int(similarities.argmax()) == target_style_id),
    }
