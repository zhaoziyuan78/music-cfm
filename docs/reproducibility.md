# Reproducibility

Run directories persist resolved Hydra configuration, Python/platform information, and git state.
Each trainer atomically overwrites one rolling `last.pt` checkpoint containing model/EMA,
optimizer, scheduler, gradient scaler, global step/epoch/batch cursor, worker count,
Python/NumPy/PyTorch RNG state, provenance hashes, the condition schema, and the deterministic
Stage-2 projector seed rules. Dynamic projection matrices are regenerated and are not stored. Use
`resume=true` to discover it automatically. Changing between one and four workers preserves the
optimizer step and resets only the incompatible within-epoch cursor. Latent statistics are
computed per latent-token position only from the train split; validation/test loads reject
provenance or normalization-schema mismatches.

Use seed 0, 1, and 2 for ambiguity analysis. Compare CFM and DDIM at 16, 32, and 64 solver steps and report actual NFE. Determinism can vary across CUDA kernels even with fixed seeds; exact hardware and package versions should accompany final results.
