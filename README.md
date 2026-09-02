# CFMusic: Shared-Noise Conditional Transport

This repository implements **Shared-Noise Conditional Transport for Unpaired Counterfactual Music Style Transfer**. It studies whether Conditional Flow Matching (CFM) and deterministic DDIM can approximately abduct style-independent exogenous noise from a factual MIDI latent and reuse that same noise under a new style condition.

Training is intentionally unpaired. A training item contains one source MIDI segment, its observed style/auxiliary labels, dataset identity, and metadata. It never contains a target MIDI, nearest target, same-song target, or synthetic counterfactual reference. OT-CFM couples Gaussian noise to factual latents within a style minibatch; it does not pair music with music.

The default complete-latent intervention is

\[
z=E_\psi(x),\qquad u=F_\theta^{-1}(z;s),\qquad
z^{cf}=F_\theta(u;s'),\qquad x^{cf}=D_\psi(z^{cf}).
\]

All styles share one conditional backbone. Classifier-free guidance and condition dropout are disabled because abduction and prediction must use the same conditional mechanism.

## Installation

Python 3.11 and PyTorch 2.x are supported. Create the locked environment with:

```bash
uv sync --extra dev --extra ot
```

All paths are Hydra-overridable and have environment-variable equivalents. A cluster layout can be selected without editing code:

```bash
export CFMUSIC_PROJECT_ROOT=/home/gus.xia/ziyuan/music-scm
export CFMUSIC_DATA_ROOT=/l/users/gus.xia/ziyuan/music-scm/data
export CFMUSIC_CHECKPOINTS_DIR=/l/users/gus.xia/ziyuan/music-scm/checkpoints
export CFMUSIC_RUNS_DIR=/l/users/gus.xia/ziyuan/music-scm/runs
export CFMUSIC_ARTIFACTS_DIR=/l/users/gus.xia/ziyuan/music-scm/artifacts
```

Equivalent overrides such as `paths.data_root=/absolute/path` work on every command. Each Hydra run records the resolved config, runtime environment, and git state.

Long-running downloads, extraction, preprocessing, training, latent caching, generation, and
evaluation render progress, throughput, ETA, and stage-specific statistics. Progress remains
enabled in redirected Slurm logs, and distributed jobs only render it on rank zero. Set
`CFMUSIC_PROGRESS=0` to disable all project progress bars.

## Datasets and licenses

| Dataset | Role | License gate |
|---|---|---|
| XMIDI | large-scale genre/emotion/factorial experiments | `UNKNOWN_VERIFY_WITH_DATASET_AUTHORS` |
| EMOPIA | piano 4Q emotion | CC-BY-NC-SA-4.0, non-commercial |
| VGMIDI | 4Q emotion and domain shift | `UNKNOWN_VERIFY_WITH_DATASET_AUTHORS` |
| Groove MIDI Dataset | top-8 primary drum styles | CC-BY-4.0 |

Review [docs/licenses.md](docs/licenses.md) before accepting a license. Unknown licenses additionally require explicit acknowledgement:

```bash
uv run python -m cfmusic.commands.download \
  datasets='[xmidi,emopia,vgmidi,groove]' \
  paths.data_root=/l/users/gus.xia/ziyuan/music-scm/data \
  license.accept=true license.acknowledge_unknown=true
```

Use `download.dry_run=true` to validate gates and destinations without network writes. Downloads are locked, resumable where the source permits it, checksum-verified, atomically completed, and safely extracted.

## Preprocessing

Each dataset has a separate adapter. Preprocessing audits MIDI structure, canonicalizes events, groups exact duplicates, performs leakage-safe group splitting, creates bar-aligned segments, and writes a Parquet manifest:

```bash
# XMIDI only (the default first experiment)
bash preprocess.sh

# Later datasets can also be prepared independently
uv run python -m cfmusic.commands.prepare data=xmidi paths.data_root=$CFMUSIC_DATA_ROOT
uv run python -m cfmusic.commands.prepare data=emopia paths.data_root=$CFMUSIC_DATA_ROOT
uv run python -m cfmusic.commands.prepare data=vgmidi paths.data_root=$CFMUSIC_DATA_ROOT
uv run python -m cfmusic.commands.prepare data=groove paths.data_root=$CFMUSIC_DATA_ROOT
uv run python -m cfmusic.commands.audit datasets='[xmidi,emopia,vgmidi,groove]'
```

`bash preprocess.sh emopia`, `bash preprocess.sh vgmidi`, and `bash preprocess.sh groove` run the
same isolated prepare-and-audit pipeline for the other datasets.

EMOPIA splits by song/YouTube identity, VGMIDI by series+game+piece, Groove uses official splits,
and XMIDI keeps canonical duplicates together. MIDI now uses the deterministic tokenizer from
[BEAT-code](https://github.com/Lekai-Qian/BEAT-code): its fixed 593-token vocabulary represents
each beat as four base-3 onset/sustain/silence steps (`PAT`), with `PIT`, `INS`, `BEAT`, `BAR`,
tempo, time-signature, rest, and drum tokens. The project-native adapter reads MIDI directly, so
it does not require a BEAT model, checkpoint, dependency, or separate tokenizer training stage.
Unlike the reference multitrack conversion, this adapter retains the source MIDI's actual velocity
and all 128 program IDs to improve reconstruction fidelity.

The manifest stores the exact untruncated BEAT length. The pitched model accepts 2560 tokens and
the drum model 512; overlength segments are excluded rather than silently training against a
truncated target. Because old manifests and codec checkpoints use a different vocabulary, rerun
preparation for each dataset before training its codec and start a fresh codec run
(`resume=false`). For the first XMIDI-only experiment, `bash preprocess.sh` is sufficient; EMOPIA,
VGMIDI, and Groove do not need to be prepared yet. Old codec checkpoints and latent caches cannot
be reused with BEAT. A deterministic tokenizer-only audit can be run before training:

```bash
uv run python scripts/evaluate_beat_tokenizer.py \
  --data-root "$CFMUSIC_DATA_ROOT" --dataset xmidi --samples 1000 \
  --max-sequence-length 2560 --output reports/beat_tokenizer_xmidi.json
```

The checked-in 500-segment audits report 100% structural encode/decode/encode identity for all
four datasets. Groove is also 100% token-identical; pitched token accuracy is 93.3%--99.1%, with
only `VEL` tokens changing when BEAT's one velocity per pitch-pattern must summarize overlapping
same-pitch notes (mean velocity error 0.06--0.38 on the 0--127 scale). Note timing, duration,
instrument, pitch, pattern, bar structure, and drum events remain identical at tokenizer level.

Preprocessing reads and parses each MIDI only once while computing both required hashes and all
segment statistics. It uses up to eight CPUs available to the job by default. Override with
`preprocessing.workers=1` for serial execution or another explicit worker count; tune task
dispatch with `preprocessing.worker_chunksize=32`. Manifest rows are streamed to Parquet in
batches instead of being retained as a second full in-memory table.

## Training the VAE and caching latents

Every PyTorch training entry point supports either one GPU or synchronous four-GPU DDP. Batch
sizes in the YAML files are **per GPU**. The examples below pin a one-node job explicitly.

XMIDI, EMOPIA, and VGMIDI use independent codec experiments. To train XMIDI first:

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e00_xmidi_codec tokenizer=beat resume=false \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_codec experiment=e00_xmidi_codec \
  tokenizer=beat resume=false paths.data_root=$CFMUSIC_DATA_ROOT
```

The checked-in Slurm launcher also defaults to XMIDI:

```bash
sbatch vae.sh
# Resume the isolated XMIDI checkpoint after preemption:
sbatch --export=ALL,CFMUSIC_CODEC_RESUME=true vae.sh
```

Train EMOPIA independently:

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e01_emopia_codec tokenizer=beat resume=false \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_codec experiment=e01_emopia_codec \
  tokenizer=beat resume=false paths.data_root=$CFMUSIC_DATA_ROOT
```

Train VGMIDI independently:

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e02_vgmidi_codec tokenizer=beat resume=false \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_codec experiment=e02_vgmidi_codec \
  tokenizer=beat resume=false paths.data_root=$CFMUSIC_DATA_ROOT
```

The Groove drum codec remains separate:

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e03_groove_codec tokenizer=beat resume=false paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_codec experiment=e03_groove_codec \
  tokenizer=beat resume=false paths.data_root=$CFMUSIC_DATA_ROOT
```

The four rolling checkpoints are isolated at
`e00_xmidi_codec/codec/xmidi/last.pt`, `e01_emopia_codec/codec/emopia/last.pt`,
`e02_vgmidi_codec/codec/vgmidi/last.pt`, and `e03_groove_codec/codec/groove/last.pt` below the
configured checkpoint root. No training sampler or checkpoint is shared between datasets.
Training scalars are appended to `metrics.jsonl`/`metrics.csv`, while periodic held-out
reconstruction and latent-reliance metrics are written under each codec directory in
`validation/metrics.jsonl` and `validation/metrics.csv`.

All codec profiles apply decoder-token dropout to reduce the gap between
teacher-forced loss and autoregressive reconstruction. The pitched BEAT VAE is enlarged to
`d_model=768`, 12 encoder + 12 decoder layers, 12 heads, and a `64 x 512` latent; the Groove model
uses `d_model=512`, 8 + 8 layers, and a `32 x 256` latent. KL weight is reduced to `1e-4` with a
long warmup so posterior collapse does not sacrifice reconstruction. Periodic validation now also
runs greedy free decoding at each sample's exact bar count and logs timing/instrument-aware
note-event F1, not only teacher-forced cross-entropy. The two small pitched datasets use their own
200-epoch caps and shorter LR/KL warmups instead of inheriting XMIDI's large-corpus schedule.

After training, audit a checkpoint on a deterministic validation subset (the report compares raw
and EMA weights and includes free-running note-event F1):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate_codec_checkpoint.py \
  --checkpoint "$CFMUSIC_CHECKPOINTS_DIR/e00_xmidi_codec/codec/xmidi/last.pt" \
  --data-root "$CFMUSIC_DATA_ROOT" --dataset xmidi --samples 256 \
  --generation-samples 8 --generation-max-length 2560 \
  --output reports/codec_evaluation/xmidi.json
```

Transport always uses the frozen posterior mean. Cache it with train-only feature statistics and provenance hashes:

The pitched CFM, OT-CFM, DDIM, split-transport, Stage-2, and generation profiles all consume the
current `64 x 512` BEAT latent. Old `32 x 256` caches and transport checkpoints are intentionally
rejected before training. Groove keeps its independent `32 x 256` profile.

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.cache_latents data=xmidi \
  codec_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e00_xmidi_codec/codec/xmidi/last.pt \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s (recommended for XMIDI)
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.cache_latents data=xmidi \
  codec_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e00_xmidi_codec/codec/xmidi/last.pt \
  paths.data_root=$CFMUSIC_DATA_ROOT
```

Each rank receives a non-overlapping contiguous manifest partition and writes its own shard
subdirectory. Rank zero computes the checkpoint hash once, merges the indexes, streams train-only
normalization statistics, and atomically publishes the completed cache. The prior cache remains
usable until the new one is complete. On four A100 40G GPUs the default encoder batch is 384 per
GPU (about 23.3 GiB in a worst-case 2560-token test), with 32 total data-loader workers. The cache
uses the same overlength exclusion as VAE training and records codec/tokenizer/manifest hashes,
weight variant, latent shape, dtype, and normalization hash. XMIDI's roughly 1.93 million valid
segments require approximately 118 GiB for raw FP16 latent tensors, plus indexes and serialization
overhead.

## CFM, DDIM, and abduction fine-tuning

Stage-1 transport training:

```bash
# One A100: shared I-CFM
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e20_cfm_base paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s: shared I-CFM
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_transport experiment=e20_cfm_base \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Conditional DDIM uses the same launcher; choose e10_ddim_vanilla or e11_ddim_fpi
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e10_ddim_vanilla paths.data_root=$CFMUSIC_DATA_ROOT
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_transport experiment=e11_ddim_fpi \
  paths.data_root=$CFMUSIC_DATA_ROOT
```

The XMIDI CFM backbone is a 768-wide, 10-layer AdaLN DiT. Its primary objective remains ordinary
independent flow matching. A second forward pass over 128 examples per GPU compares the factual
velocity target under the correct style and an observed, in-support wrong style. A margin loss
focuses on the low-time/high-noise endpoint where the state carries the least factual style
information. Its weight is 0.5 and its required MSE gap is 0.10, so it remains active longer
without overtaking the primary CFM objective. `condition_gap`, `condition_accuracy`,
`condition_correct_error`, and `condition_wrong_error` are written to every training log. The main
loss is also weighted by the square root of inverse class frequency, which raises the contribution
of XMIDI's smallest style without repeatedly loading or heavily oversampling its examples.

The current `64 x 512` profiles were tested on four A100 40G GPUs with the configured per-GPU
batches. The enlarged CFM Stage-1 peaks at 20.03 GiB/GPU and sustains about 6,820 samples/second
globally. Its differentiable Stage-2 worst path peaks at 33.00 GiB/GPU. The unchanged DDIM
Stage-1/DDIM-FPI Stage-2 profiles remain near 8.3/20.0 GiB. These are complete
forward/backward/optimizer measurements, including the Stage-2 four-step inverse and round trip.

Stage-2 HSIC + classwise-prior + round-trip fine-tuning:

```bash
# One A100
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.finetune_abduction \
  experiment=e22_cfm_exoreg transport_checkpoint=/path/to/stage1/last.pt \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.finetune_abduction experiment=e22_cfm_exoreg \
  transport_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e20_cfm_base/transport_stage1/last.pt paths.data_root=$CFMUSIC_DATA_ROOT
```

Stage-2 initializes from the Stage-1 EMA parameters, runs the differentiable abduction objective
every two steps, and continues the condition-margin objective. Its HSIC, classwise-prior, and
round-trip coefficients are scaled so their combined contribution is visible rather than being
lost below roughly 0.2% of the base loss. Final generation also selects the Stage-2 EMA parameters
by default; set `counterfactual.transport_weights=raw` only for an explicit raw-vs-EMA ablation.
At every rolling checkpoint, both stages also evaluate a fixed 96-latent, six-style validation
probe against all five wrong styles with identical noise and time draws. The resulting
`validation_condition_*` fields distinguish genuine held-out condition following from merely
satisfying the margin on current training batches.

The full XMIDI continuation can be submitted with one Slurm command:

```bash
sbatch cfm.sh

# Typical restart after an unrelated stage has already completed
sbatch --export=ALL,CFMUSIC_SKIP_LATENT_CACHE=true,CFMUSIC_CFM_RESUME=true cfm.sh
```

The enlarged CFM checkpoint is intentionally architecture-incompatible with older 512-wide,
8-layer CFM checkpoints. Start this revised Stage-1/Stage-2 pair with
`CFMUSIC_CFM_RESUME=false`; use `true` only after the revised run has written its own `last.pt`.

All torch trainers atomically overwrite one intermediate checkpoint named `last.pt`; they no
longer retain `step-XXXXXXXX.pt` copies, and the first new save removes legacy step checkpoints in
that run directory. Set `resume=true` to discover `last.pt` in the command's normal output
directory and restore model, optimizer, scheduler, AMP scaler, EMA, step/epoch/data cursor, and RNG
state. `resume_from=/explicit/checkpoint.pt` remains available when the checkpoint is elsewhere.
For example:

```bash
# Continue the normal Stage-1 run; this works with either launcher above
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e20_cfm_base resume=true paths.data_root=$CFMUSIC_DATA_ROOT

# Change the rolling interval if desired
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e00_xmidi_codec resume=true codec.training.checkpoint_interval=1000 \
  paths.data_root=$CFMUSIC_DATA_ROOT
```

Only rank zero writes progress, metrics, and checkpoints. Switching between one and four GPUs on
resume preserves the optimizer step and safely resets only the within-epoch data cursor. The
default profiles target a 40 GiB A100. The long-sequence pitched codec retains activation
checkpointing and peaks near 30 GiB at batch 32. Stage-1 transport does not checkpoint
activations: the enlarged CFM processes 512 samples per GPU at 20.03 GiB, while the unchanged
DDIM profile remains near 8.3 GiB. The balanced Stage-2 batch of 64 peaks at 33.00 GiB for the
enlarged CFM and about 20.0 GiB for DDIM-FPI; its batch composition is kept fixed because the
independence losses depend on the number of examples per class. AdamW uses its fused CUDA
implementation,
static buffers are not re-broadcast on every DDP forward, and EMA is updated in equivalent ten-step
chunks. With four GPUs, the global effective batch is the configured per-GPU batch times four (and
times gradient accumulation).

Latent transport training uses `sdpa_backend: math` by default for both Stage 1 and Stage 2 while
retaining BF16 autocast for the rest of the model. This avoids the severely amplified BF16 fused
SDPA/Flash-Attention backward gradients observed in trained AdaLN blocks on A100. The latent
sequence is only 64 tokens, so the math attention matrix remains small. Codec and token-evaluator
training keep their automatic attention backend because their sequences can reach 2560 and 2048
tokens, respectively.
The startup plan prints the selected backend; `sdpa_backend=math` should be visible before a new
transport run begins.

Every training stage writes the same observable logging bundle in its checkpoint directory:

- `metrics.jsonl` and `metrics.csv`: machine-readable scalar history;
- `training.log`: compact one-line-per-record text log;
- `training_curves.png`: an atomically replaced dashboard containing raw and smoothed curves;
- `tensorboard/events.out.tfevents.*`: TensorBoard scalars;
- `last.pt` (or `last.joblib` for the descriptor baseline): the rolling resume checkpoint.

Codec validation writes the same bundle under `validation/`. The descriptor baseline writes its
bundle under `descriptor_mlp_training/`, so it cannot collide with the Transformer evaluator.
The post-hoc temporal leakage probe writes its bundle under
`artifacts/<experiment>/<dataset>/evaluation/temporal_probe_training/`.
Curves are refreshed periodically during training and once more on clean shutdown. A fresh run
resets the old scalar logs and TensorBoard events; `resume=true` restores and extends both the
history and curve. For example, inspect a Stage-1 run with:

```bash
uv run tensorboard --logdir \
  /l/users/gus.xia/ziyuan/music-scm/checkpoints/e20_cfm_base/transport_stage1/tensorboard
```

Latent transport loaders shuffle shards and samples within each shard, then assign complete shard
runs evenly to ranks. This avoids reloading a roughly 128 MiB latent file for nearly every random
sample. Stage-2 style-balanced batches select one shard per requested style and use a small mmap LRU
cache. On the current four-A100 node, the revised XMIDI CFM short run sustained about 0.30
seconds/step and 6.82k samples/second. The Stage-1 budget is capped at 60 data epochs, 50k optimizer
steps, and a 5k-step floor; the current four-GPU XMIDI cache therefore schedules about 45.3k steps,
or roughly 3.8 hours of pure optimizer time at the measured short-run rate. Stage-2 uses 12k steps
and a four-step differentiable training solver; final generation and evaluation still use the
configured 32-step solver.

Latent caching has its own inference batch size, and all inference paths use inference mode plus
bf16 where numerically safe. Length bucketing avoids padding every codec or evaluator batch to its
maximum token length, while persistent loader workers overlap MIDI parsing with GPU work. The token
evaluator uses one factual segment per song, as its pre-existing de-duplication intended, instead
of silently training on roughly twenty overlapping XMIDI windows per song.

Validate a node with the full-size memory suite before a long run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/memory_smoke.py \
  --cases codec_train,codec_encode,codec_decode --codec-profile transformer_vae \
  --codec-batch 32 --codec-tokens 2560 --limit-gib 38 \
  --output reports/beat_codec_a100_memory.json
```

Every training progress bar and metrics log includes peak allocated GPU memory and the downstream
trainers also report measured step time and global sample throughput. Worst-length synthetic A100
measurements selected pitched batch 32 with activation checkpointing (29.2 GiB) and Groove batch 64
without it (16.0 GiB); these preserve the previous effective batches while eliminating eight and
two serial micro-batches per optimizer step. The pitched codec is capped at eight data epochs and
100k optimizer steps.

## Evaluators, generation, and evaluation

Evaluators train only on real train-split MIDI:

```bash
# One A100: token Transformer evaluator
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_evaluator \
  data=xmidi evaluator=transformer task=genre paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s: token Transformer evaluator
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_evaluator data=xmidi evaluator=transformer task=genre \
  paths.data_root=$CFMUSIC_DATA_ROOT

# CPU-only scikit-learn descriptor baseline (not a GPU-distributed stage)
uv run python -m cfmusic.commands.train_evaluator data=xmidi evaluator=descriptor_mlp task=genre
```

The Transformer evaluator uses the same `last.pt`/`resume=true` contract. The descriptor MLP
atomically overwrites `last.joblib` every `evaluator.checkpoint_interval` iterations and also
supports `resume=true`; its final export remains `descriptor_mlp.joblib`.

Generate and evaluate unpaired counterfactuals:

```bash
uv run python -m cfmusic.commands.generate_counterfactuals \
  experiment=e22_cfm_exoreg transport_checkpoint=/l/users/gus.xia/ziyuan/music-scm/checkpoints/e22_cfm_exoreg/transport_stage2/last.pt \
  codec_checkpoint=/l/users/gus.xia/ziyuan/music-scm/checkpoints/e00_xmidi_codec/codec/xmidi/last.pt counterfactual.target_policy=all_other

# Four A100s: split the selected sources across four independent generation workers
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.generate_counterfactuals experiment=e22_cfm_exoreg \
  transport_checkpoint=/l/users/gus.xia/ziyuan/music-scm/checkpoints/e22_cfm_exoreg/transport_stage2/last.pt \
  codec_checkpoint=/l/users/gus.xia/ziyuan/music-scm/checkpoints/e00_xmidi_codec/codec/xmidi/last.pt \
  counterfactual.target_policy=all_other

uv run python -m cfmusic.commands.evaluate experiment=e22_cfm_exoreg
uv run python -m cfmusic.commands.build_report \
  report.experiments='[e10_ddim_vanilla,e11_ddim_fpi,e20_cfm_base,e22_cfm_exoreg]'
```

Generation now selects at most 10 unique songs per source style (60 sources / 300 ordered
transitions for six-style XMIDI) with vectorized grouping, and reads only those rows from the large
MIDI manifest. Each source is inverted and reconstructed once, all target conditions are
transported and decoded in GPU batches, and codec decoding uses projected self/cross-attention K/V
caches. In four-GPU jobs, only rank zero hashes the large codec checkpoint. A worst-case A100 test
decoded six full 2560-token sequences in 22.7 seconds at 1.32 GiB allocated memory; ordinary runs
can finish sooner at EOS. Completed artifacts are skipped on reruns. Override
`counterfactual.max_sources_per_style`, `counterfactual.max_total_sources`, or
`counterfactual.targets_per_source` only when a larger evaluation set is needed.

Evaluation reports factual round trips, generated-noise recovery, leakage probes, style probability, non-intervened retention, separate symbolic preservation metrics, counterfactual algebra, distribution distances, and cross-seed spread. It never requires a paired target MIDI.

## Experiment matrix

P0 configurations are E00 codec ceiling; E10/E11 DDIM; E20/E22 shared CFM; E30 XMIDI factorial; E31 EMOPIA; E32 VGMIDI; and E33 Groove. P1 adds E23 OT-CFM, E40 weak conserved/editable split, E50 independent per-style flows, E51 shuffled labels, and E60 joint 4Q domain training. Every entry is under `configs/experiment/`.

## Reproducibility and checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Seeds, RNG states, style vocabulary/provenance hashes, normalization, optimizer/scheduler/scaler state, EMA, and fixed random projection are checkpointed. Matched DDIM/CFM evaluation uses 16/32/64-step grids and counts every model evaluation, including FPI calls. See [docs/reproducibility.md](docs/reproducibility.md).

## Known limitations

Unpaired observational labels do not identify a unique individual counterfactual. MIDI quantization and VAE reconstruction impose a codec ceiling; symbolic evaluators can be biased; XMIDI and VGMIDI licenses need author verification; EMOPIA is non-commercial; full XMIDI training is computationally expensive; and FluidSynth rendering requires a user-supplied SoundFont.

Low style predictability from abducted noise is evidence of distributional
exogeneity, not proof of unique individual-level counterfactual identification.
