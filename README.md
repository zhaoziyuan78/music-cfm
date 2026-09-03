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

Transport always uses the frozen posterior mean. Cache it with train-only **per-token** feature
statistics and provenance hashes:

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
weight variant, latent shape, dtype, normalization schema, and normalization hash. Mean and
standard deviation have shape `64 x 512`: each learned VAE query position is normalized
independently rather than pooling the 64 non-exchangeable tokens into one distribution. Caches
without `normalization_schema_version=per-token-v2` are rejected. XMIDI's roughly 1.93 million valid
segments require approximately 118 GiB for raw FP16 latent tensors, plus indexes and serialization
overhead.

Four-GPU caching partitions each split by estimated quadratic attention cost rather than only by
row count. Long NFS preparation, rank stragglers, index merging, and atomic publication synchronize
through build-scoped filesystem markers, so they do not hold an NCCL barrier open past its watchdog
timeout. `latent_cache.synchronization_timeout_seconds` bounds a genuinely stalled build and
defaults to two hours.

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

The XMIDI CFM backbone is a 768-wide, 10-layer AdaLN DiT. Its objective is ordinary independent
flow matching. The former wrong-condition margin has been removed: a factual flow path paired
with an incorrect class is not a valid conditional-FM training pair. Conditional endpoint quality
is instead measured by class-conditional MMD and sliced Wasserstein distance between
`F_s(epsilon)` and held-out factual latents. The main loss remains weighted by the square root of
inverse class frequency.

All condition construction now goes through schema `task-aware-v2`. A genre run activates
`dataset + task + style=genre` and sets `genre_id=emotion_id=None`; an emotion run similarly puts
only emotion in the style slot. A factorial run uses a constant style sentinel and activates only
the separate genre and emotion slots. A factorial intervention changes exactly one of those axes.
The schema and task are checkpoint provenance, and resume, Stage-2 initialization, and generation
reject older checkpoints rather than silently applying incompatible condition semantics.

The `64 x 512` backbone and batch profiles were measured on four A100 40G GPUs. The enlarged CFM
Stage 1 peaks at 20.03 GiB/GPU and sustains about 6,820 samples/second globally. The four-step
differentiable Stage-2 solver/round-trip path measured 33.00 GiB/GPU before this audit; its new
dynamic projection and gathered-feature tensors add less than 0.1 GiB/GPU by construction, leaving
over 6 GiB of device headroom. The unchanged DDIM Stage-1/DDIM-FPI Stage-2 profiles remain near
8.3/20.0 GiB. Confirm the first logged peak on the target driver/PyTorch build before a long run.

Stage-2 multi-view exogeneity + round-trip fine-tuning:

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

Stage-2 initializes from the Stage-1 EMA parameters and runs differentiable abduction every two
steps. It uses three independently resampled Rademacher projections, token-wise mean/std, and a
random token-by-channel block. HSIC, standard-normal sliced Wasserstein, and cross-class MMD are
computed across those views. Projectors change every step, while validation uses a disjoint seed
stream. In DDP, differentiable all-gather forms the regularizers on the complete four-GPU batch.
The balanced sampler is hierarchical (`style -> unique sample_id -> one segment`), so one song can
appear at most once per class in a batch; Stage 1 likewise chooses one changing segment per song
pass while retaining shard-local reads and its original optimizer budget.

Stage-2 EMA decay is configurable and defaults to 0.999. At each rolling checkpoint, raw and EMA
weights are both evaluated for endpoint MMD/SWD, held-out noise HSIC/SWD/cross-class MMD, and
round-trip loss. Generation defaults to raw weights; select `counterfactual.transport_weights=ema`
only after comparing these validation fields.

The full XMIDI continuation can be submitted with one Slurm command:

```bash
sbatch cfm.sh

# Typical restart after an unrelated stage has already completed
sbatch --export=ALL,CFMUSIC_SKIP_LATENT_CACHE=true,CFMUSIC_CFM_RESUME=true cfm.sh
```

The new per-token normalization and condition schema make every earlier E20/E22 checkpoint and
latent cache intentionally incompatible. Rebuild XMIDI latents, then start Stage 1 and Stage 2
with `CFMUSIC_CFM_RESUME=false`; use `true` only after the revised run has written its own
`last.pt`. The VAE checkpoint itself remains valid.

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
DDIM profile remains near 8.3 GiB. The balanced Stage-2 batch stays at 64 per GPU; its dominant CFM
solver path measured 33.00 GiB before the sub-0.1-GiB dynamic-view addition, while DDIM-FPI is about
20.0 GiB. Its batch composition is kept fixed because the independence losses depend on examples
per class. AdamW uses its fused CUDA implementation, dynamic projection matrices are never retained
in checkpoints, and EMA is updated in equivalent ten-step chunks. With four GPUs, the global
effective batch is the configured per-GPU batch times four (and times gradient accumulation).

Latent transport training uses `sdpa_backend: math` by default for both Stage 1 and Stage 2 while
retaining BF16 autocast for the rest of the model. This avoids the severely amplified BF16 fused
SDPA/Flash-Attention backward gradients observed in trained AdaLN blocks on A100. The latent
sequence is only 64 tokens, so the math attention matrix remains small. Codec training keeps its
automatic attention backend because its sequences can reach 2560 tokens.
The startup plan prints the selected backend; `sdpa_backend=math` should be visible before a new
transport run begins.

Every training stage writes the same observable logging bundle in its checkpoint directory:

- `metrics.jsonl` and `metrics.csv`: machine-readable scalar history;
- `training.log`: compact one-line-per-record text log;
- `training_curves.png`: an atomically replaced dashboard containing raw and smoothed curves;
- `tensorboard/events.out.tfevents.*`: TensorBoard scalars;
- `last.pt`: the rolling resume checkpoint.

Codec validation writes the same bundle under `validation/`. The post-hoc temporal leakage probe writes its bundle under
`artifacts/<experiment>/<dataset>/evaluation/temporal_probe_training/`.
Curves are refreshed periodically during training and once more on clean shutdown. A fresh run
resets the old scalar logs and TensorBoard events; `resume=true` restores and extends both the
history and curve. For example, inspect a Stage-1 run with:

```bash
uv run tensorboard --logdir \
  /l/users/gus.xia/ziyuan/music-scm/checkpoints/e20_cfm_base/transport_stage1/tensorboard
```

Latent transport loaders retain locality by visiting only one or two contiguous shards per batch.
Within that locality, Stage 1 weights songs equally and selects a changing segment from each song;
Stage 2 balances styles and draws unique songs before selecting one segment. The Stage-1 budget is
capped at 60 data passes, 50k optimizer steps, and a 5k-step floor. Stage 2 uses 12k steps and a
four-step differentiable training solver; final generation uses the configured 32-step solver.

Latent caching has its own inference batch size, and all inference paths use inference mode plus
bf16 where numerically safe. Length bucketing avoids padding every codec batch to its maximum token
length, while persistent loader workers overlap MIDI parsing with GPU work.

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

## CLaMP 2, generation, and evaluation

The main style metric no longer trains an in-domain Transformer classifier. It uses the official
[CLaMP 2](https://github.com/sanderwood/clamp2) music/text embedding model and fixed prompts such
as `This is a piece of rock music.`. This checkout is installed under `external/clamp2`; its two
large released checkpoints live under `/l/users/gus.xia/ziyuan/music-scm/checkpoints/clamp2` and
are symlinked into `code/`, as required by the upstream extractor. CLaMP 2's runtime dependencies
are part of the project environment, so no separate Conda environment is needed:

```bash
git clone https://github.com/sanderwood/clamp2 external/clamp2
export CLAMP2_REPOSITORY=$CFMUSIC_PROJECT_ROOT/external/clamp2
export CLAMP2_CACHE_DIR=/l/users/gus.xia/ziyuan/music-scm/checkpoints/clamp2/huggingface
```

Evaluation converts each generated MIDI to CLaMP 2's lossless MTF input, extracts normalized MIDI
and style-text embeddings once, and reports target similarity, source similarity, their margin,
and zero-shot target success. The prompt is configurable through
`evaluation.clamp2.style_template`.

The evaluation launched by `cfm.sh` fixes `task=genre`, so CLaMP 2 compares only the XMIDI style
(genre) labels and does not treat emotion or another auxiliary label as an intervention. The
noise-leakage probes, content-preservation metrics, and MIDI-quality diagnostics are still run.

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

uv run python -m cfmusic.commands.evaluate experiment=e22_cfm_exoreg \
  evaluation.clamp2.repository=$CLAMP2_REPOSITORY
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

Evaluation reports CLaMP 2 zero-shot style alignment, pitch-class histogram cosine, a
transposition-invariant melody-contour cosine, descriptor/tempo/density preservation, independent
linear/MLP/temporal noise probes, and reference-free MIDI validity/range/density/duration metrics.
It never requires a paired target MIDI.

## Experiment matrix

P0 configurations are E00 codec ceiling; E10/E11 DDIM; E20/E22 shared CFM; E30 XMIDI factorial; E31 EMOPIA; E32 VGMIDI; and E33 Groove. P1 adds E23 OT-CFM, E40 weak conserved/editable split, E50 independent per-style flows, E51 shuffled labels, and E60 joint 4Q domain training. Every entry is under `configs/experiment/`.

## Reproducibility and checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Seeds, RNG states, style vocabulary/provenance hashes, normalization schema,
optimizer/scheduler/scaler state, EMA, condition schema, and deterministic dynamic-projector seed
rules are recorded. Matched DDIM/CFM evaluation counts every model evaluation, including FPI
calls. See [docs/reproducibility.md](docs/reproducibility.md).

## Known limitations

Unpaired observational labels do not identify a unique individual counterfactual. MIDI quantization and VAE reconstruction impose a codec ceiling; symbolic evaluators can be biased; XMIDI and VGMIDI licenses need author verification; EMOPIA is non-commercial; full XMIDI training is computationally expensive; and FluidSynth rendering requires a user-supplied SoundFont.

Low style predictability from abducted noise is evidence of distributional
exogeneity, not proof of unique individual-level counterfactual identification.
