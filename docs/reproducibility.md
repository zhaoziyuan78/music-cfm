# Reproducibility

Run directories persist resolved Hydra configuration, Python/platform information, and git state.
Each trainer atomically overwrites one rolling `last.pt` checkpoint containing model/EMA,
optimizer, scheduler, gradient scaler, global step/epoch/batch cursor, worker count,
Python/NumPy/PyTorch RNG state, provenance hashes, and the Stage-2 fixed projection. Use
`resume=true` to discover it automatically. Changing between one and four workers preserves the
optimizer step and resets only the incompatible within-epoch cursor. Latent statistics are
computed only from the train split and validation/test loads reject provenance mismatches.

Use seed 0, 1, and 2 for ambiguity analysis. Compare CFM and DDIM at 16, 32, and 64 solver steps and report actual NFE. Determinism can vary across CUDA kernels even with fixed seeds; exact hardware and package versions should accompany final results.
