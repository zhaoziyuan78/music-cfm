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

The pitched CFM, OT-CFM, DDIM, split-transport, unified round-trip, and generation profiles all consume the
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

## Unified CFM training and DDIM ablations

The active XMIDI method uses one CFM training run. It does not initialize or launch a separate
abduction-fine-tuning stage. The relabeled latent cache remains unchanged and is selected through
`data.latent_index`:

```bash
# One A100: unified conditional OT-CFM (the overlay must already exist)
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  data.latent_index=$CFMUSIC_DATA_ROOT/latents/xmidi/index_clamp2_segment_v2.parquet \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Four A100s: recommended
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_transport experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  data.latent_index=$CFMUSIC_DATA_ROOT/latents/xmidi/index_clamp2_segment_v2.parquet \
  paths.data_root=$CFMUSIC_DATA_ROOT

# Conditional DDIM uses the same launcher; choose e10_ddim_vanilla or e11_ddim_fpi
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e10_ddim_vanilla paths.data_root=$CFMUSIC_DATA_ROOT
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.train_transport experiment=e11_ddim_fpi \
  paths.data_root=$CFMUSIC_DATA_ROOT
```

The XMIDI CFM backbone remains a 768-wide, 10-layer AdaLN DiT, so the existing `64 x 512` VAE
latents are consumed unchanged. Training uses within-style Hungarian minibatch OT in a fixed
128-dimensional projection and samples time from `0.5 U(0,1) + 0.5 Beta(0.5,1)`. Each per-GPU
batch has 64 unique songs from each of the six pseudo-genres (384 total); rows are ordered by shard
for mmap throughput, and prompt-margin confidence weights enter the per-sample CFM loss.

Classifier-free guidance is trained with 10% per-example condition dropout. Source abduction and
same-style reconstruction are fixed at guidance 1.0. Target prediction defaults to
`v_null + 2(v_target-v_null) - 0.5(v_source-v_null)`; all required branches are concatenated into
one vector-field call. The prediction and source-repulsion scales are inference-only and may be
changed without invalidating a checkpoint.

After a 5,000-step warm-up and 5,000-step ramp, three differentiable auxiliaries run on different
offsets of an eight-step cycle: round trip at weight 0.05 (16 examples/GPU), conditional endpoint
MMD+SWD at weight 0.05 (4 examples/style/GPU), and lightweight projected exogeneity losses
(HSIC, cross-class MMD/SWD, and Gaussian-prior SWD; 4 examples/style/GPU). Every auxiliary uses an
unguided four-step solve. Staggering them prevents their autograd graphs from sharing one memory
peak with the full 384-example CFM path.

All condition construction now goes through schema `task-aware-v2`. A genre run activates
`dataset + task + style=genre` and sets `genre_id=emotion_id=None`; an emotion run similarly puts
only emotion in the style slot. A factorial run uses a constant style sentinel and activates only
the separate genre and emotion slots. A factorial intervention changes exactly one of those axes.
The schema, task, and CFG training settings are checkpoint metadata. Resume requires matching CFG
and condition-dropout settings, while generation refuses to enable guidance for a checkpoint that
was never trained with a null-condition branch. The guidance scale itself can be varied at inference.
The unified checkpoint is written to
`checkpoints/e35_otcfm_segment_cfg/transport/last.pt`.

The full XMIDI continuation can be submitted with one Slurm command:

```bash
sbatch cfm.sh

# Resume the unified CFM checkpoint and reuse the completed relabel overlay
sbatch --export=ALL,CFMUSIC_SKIP_RELABEL=true,CFMUSIC_CFM_RESUME=true cfm.sh
```

The pipeline does not alter the VAE, tokenizer, normalization, or cached latent tensors. Its
segment relabel step writes a lightweight index containing only new labels, confidence, and the
existing shard/offset references. Start the new experiment
with `CFMUSIC_CFM_RESUME=false`; set it to `true` only after this recipe has written its own
`transport/last.pt`. Earlier non-CFG transport checkpoints cannot be resumed as CFG checkpoints.

All torch trainers atomically overwrite one intermediate checkpoint named `last.pt`; they no
longer retain `step-XXXXXXXX.pt` copies, and the first new save removes legacy step checkpoints in
that run directory. Set `resume=true` to discover `last.pt` in the command's normal output
directory and restore model, optimizer, scheduler, AMP scaler, EMA, step/epoch/data cursor, and RNG
state. `resume_from=/explicit/checkpoint.pt` remains available when the checkpoint is elsewhere.
For example:

```bash
# Continue the unified CFM run; this works with either launcher above
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_transport \
  experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  data.latent_index=$CFMUSIC_DATA_ROOT/latents/xmidi/index_clamp2_segment_v2.parquet \
  resume=true paths.data_root=$CFMUSIC_DATA_ROOT

# Change the rolling interval if desired
CUDA_VISIBLE_DEVICES=0 uv run python -m cfmusic.commands.train_codec \
  experiment=e00_xmidi_codec resume=true codec.training.checkpoint_interval=1000 \
  paths.data_root=$CFMUSIC_DATA_ROOT
```

Only rank zero writes progress, metrics, and checkpoints. Switching between one and four GPUs on
resume preserves the optimizer step and safely resets only the within-epoch data cursor. The
default profiles target a 40 GiB A100. The long-sequence pitched codec retains activation
checkpointing and peaks near 30 GiB at batch 32. The CFM base path processes 512 samples per GPU;
the E25 production recipe uses 384 samples per GPU and restricts differentiable solves to the small
sparse subsets described above. CFG is not used by those training solves and does not duplicate the
full-batch CFM forward. AdamW uses its fused CUDA implementation, and EMA is updated in equivalent
ten-step chunks. With four GPUs, the main-loss global batch is 1,536.

The earlier 512-example CFM plus guided-round-trip smoke test peaked at 22.92 GiB on an A100 40G;
E25 lowers the main batch and staggers the additional objectives to retain headroom. Re-run
`scripts/memory_smoke.py --cases unified_cfm_train --limit-gib 38` after changing the model,
batch, solver, or guidance settings.

Latent transport training uses `sdpa_backend: math` by default while retaining BF16 autocast for
the rest of the model. This avoids the severely amplified BF16 fused
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
history and curve. For example, inspect the unified CFM run with:

```bash
uv run tensorboard --logdir \
  /l/users/gus.xia/ziyuan/music-scm/checkpoints/e35_otcfm_segment_cfg/transport/tensorboard
```

Latent transport loaders assign stable shard sets to ranks and keep each batch style-balanced and
song-unique. Training is capped at 50k optimizer steps. Sparse auxiliary solves use four Heun
steps; final generation uses 32 steps, guidance 1 for abduction/reconstruction, and configurable
target/source-repulsive guidance only for prediction.

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
and zero-shot target success. The new overlay selects one real cached 8-bar segment per song,
averages eight genre-prompt templates, calibrates six logit biases on validation labels, and keeps
the top 60% margin within every training pseudo-class. Evaluation reads the same prompt ensemble
and calibration from the relabel configuration.

The evaluation launched by `cfm.sh` fixes `task=genre`, so CLaMP 2 compares only the XMIDI style
(genre) labels and does not treat emotion or another auxiliary label as an intervention. The
noise-leakage probes, content-preservation metrics, and MIDI-quality diagnostics are still run.

Generate and evaluate unpaired counterfactuals:

```bash
uv run python -m cfmusic.commands.generate_counterfactuals \
  experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  data.latent_index=$CFMUSIC_DATA_ROOT/latents/xmidi/index_clamp2_segment_v2.parquet \
  transport_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e35_otcfm_segment_cfg/transport/last.pt \
  codec_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e00_xmidi_codec/codec/xmidi/last.pt \
  counterfactual.target_policy=all_other

# Four A100s: split the selected sources across four independent generation workers
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cfmusic.commands.generate_counterfactuals experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  data.latent_index=$CFMUSIC_DATA_ROOT/latents/xmidi/index_clamp2_segment_v2.parquet \
  transport_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e35_otcfm_segment_cfg/transport/last.pt \
  codec_checkpoint=$CFMUSIC_CHECKPOINTS_DIR/e00_xmidi_codec/codec/xmidi/last.pt \
  counterfactual.target_policy=all_other

uv run python -m cfmusic.commands.evaluate experiment=e25_otcfm_segment_cfg \
  experiment.name=e35_otcfm_segment_cfg \
  evaluation.clamp2.repository=$CLAMP2_REPOSITORY \
  evaluation.clamp2.relabel_config=$CFMUSIC_ARTIFACTS_DIR/diagnostics/clamp2_xmidi_segment_relabel_v2/relabel_config.json
uv run python -m cfmusic.commands.build_report \
  report.experiments='[e35_otcfm_segment_cfg]'
```

Generation now selects at most 10 unique songs per source style (60 sources / 300 ordered
transitions for six-style XMIDI) with vectorized grouping, and reads only those rows from the large
MIDI manifest. Each source is inverted and reconstructed once without CFG, all target conditions
are transported with target CFG plus source repulsion and decoded in GPU batches, and codec decoding uses projected self/cross-attention K/V
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

The active XMIDI method is E25, the single-stage balanced conditional OT-CFM recipe. E00 is the codec
ceiling; E10/E11 are DDIM ablations; and the older E20–E23 two-stage configurations are retained
only for reproducing prior results. E30 is XMIDI factorial; E31 EMOPIA; E32 VGMIDI; E33 Groove;
E40 the weak conserved/editable split; E50 independent per-style flows; E51 shuffled labels; and
E60 joint 4Q domain training. Every entry is under `configs/experiment/`.

## Reproducibility and checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Seeds, RNG states, style vocabulary/provenance hashes, normalization schema,
optimizer/scheduler/scaler state, EMA, condition schema, CFG settings, and round-trip schedule are
recorded. Matched DDIM/CFM evaluation counts every model evaluation, including CFG and FPI
calls. See [docs/reproducibility.md](docs/reproducibility.md).

## Known limitations

Unpaired observational labels do not identify a unique individual counterfactual. MIDI quantization and VAE reconstruction impose a codec ceiling; symbolic evaluators can be biased; XMIDI and VGMIDI licenses need author verification; EMOPIA is non-commercial; full XMIDI training is computationally expensive; and FluidSynth rendering requires a user-supplied SoundFont.

Low style predictability from abducted noise is evidence of distributional
exogeneity, not proof of unique individual-level counterfactual identification.
