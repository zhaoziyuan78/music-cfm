"""GPU worker for resumable, batched CLaMP 2 MIDI embedding extraction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BertConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _segments(
    patches: torch.Tensor, *, patch_length: int
) -> tuple[list[torch.Tensor], list[int]]:
    """Reproduce the official extractor's chunking and averaging weights."""

    if patches.ndim != 2 or len(patches) == 0:
        raise ValueError(f"Expected non-empty [patches, patch_size] input, got {patches.shape}")
    chunks = [patches[index : index + patch_length] for index in range(0, len(patches), patch_length)]
    chunks[-1] = patches[-patch_length:]
    full_chunks, remainder = divmod(len(patches), patch_length)
    weights = [patch_length] * full_chunks
    if remainder:
        weights.append(remainder)
    if len(weights) != len(chunks):
        raise RuntimeError("CLaMP 2 chunk/weight construction is inconsistent")
    return chunks, weights


def _load_official_model(repository: Path) -> tuple[Any, Any, int, int]:
    code_dir = repository / "code"
    sys.path.insert(0, str(code_dir))
    # The archived official repository uses top-level modules.
    from config import (  # type: ignore[import-not-found]
        CLAMP2_HIDDEN_SIZE,
        CLAMP2_LOAD_M3,
        CLAMP2_WEIGHTS_PATH,
        M3_HIDDEN_SIZE,
        PATCH_LENGTH,
        PATCH_NUM_LAYERS,
        TEXT_MODEL_NAME,
    )
    from utils import CLaMP2Model, M3Patchilizer  # type: ignore[import-not-found]

    configuration = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    device = torch.device("cuda")
    model = CLaMP2Model(
        configuration,
        text_model_name=TEXT_MODEL_NAME,
        hidden_size=CLAMP2_HIDDEN_SIZE,
        load_m3=CLAMP2_LOAD_M3,
    ).to(device)
    checkpoint = torch.load(CLAMP2_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, M3Patchilizer(), PATCH_LENGTH, M3_HIDDEN_SIZE


def _encode_file_batch(
    entries: list[dict[str, str]],
    *,
    model: Any,
    patchilizer: Any,
    patch_length: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    segments: list[torch.Tensor] = []
    owners: list[str] = []
    weights: list[int] = []
    for entry in entries:
        text = Path(entry["mtf_path"]).read_text(encoding="utf-8")
        patches = torch.tensor(patchilizer.encode(text, add_special_patches=True), dtype=torch.long)
        file_segments, file_weights = _segments(patches, patch_length=patch_length)
        segments.extend(file_segments)
        owners.extend([entry["key"]] * len(file_segments))
        weights.extend(file_weights)

    feature_sums: dict[str, np.ndarray] = {}
    weight_sums: dict[str, int] = defaultdict(int)
    for start in range(0, len(segments), batch_size):
        local_segments = segments[start : start + batch_size]
        local_owners = owners[start : start + batch_size]
        local_weights = weights[start : start + batch_size]
        patch_size = int(local_segments[0].shape[1])
        music_inputs = torch.full(
            (len(local_segments), patch_length, patch_size),
            fill_value=int(patchilizer.pad_token_id),
            dtype=torch.long,
        )
        music_masks = torch.zeros((len(local_segments), patch_length), dtype=torch.long)
        for index, segment in enumerate(local_segments):
            length = len(segment)
            music_inputs[index, :length] = segment
            music_masks[index, :length] = 1
        with torch.inference_mode():
            features = model.get_music_features(
                music_inputs.cuda(non_blocking=True),
                music_masks.cuda(non_blocking=True),
                get_normalized=True,
            )
        array = features.float().cpu().numpy()
        for owner, weight, feature in zip(
            local_owners, local_weights, array, strict=True
        ):
            if owner not in feature_sums:
                feature_sums[owner] = np.zeros_like(feature, dtype=np.float64)
            feature_sums[owner] += feature.astype(np.float64) * weight
            weight_sums[owner] += weight

    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        key = entry["key"]
        feature = (feature_sums[key] / weight_sums[key]).astype(np.float32)
        destination = output_dir / f"{key}.npy"
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, feature)
        temporary.replace(destination)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    entries = [
        json.loads(line)
        for line in args.file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pending = [entry for entry in entries if not (args.output_dir / f"{entry['key']}.npy").is_file()]
    if not pending:
        print(f"All {len(entries)} embeddings already exist")
        return
    model, patchilizer, patch_length, _ = _load_official_model(args.repository.resolve())
    for start in tqdm(range(0, len(pending), args.batch_size), desc="CLaMP 2 batched files"):
        _encode_file_batch(
            pending[start : start + args.batch_size],
            model=model,
            patchilizer=patchilizer,
            patch_length=patch_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
