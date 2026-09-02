# Data pipeline

The four adapters discover only labelled factual MIDI. Processed Parquet rows follow the schema in `cfmusic.data.schema` and contain no target/reference fields. Exact canonical duplicates are assigned one group before splitting. EMOPIA uses song/YouTube groups, VGMIDI uses series+game+piece, and Groove preserves official train/validation/test splits. Invalid files are isolated in `invalid_files.parquet` rather than terminating a full scan.

The default segmentation is eight 4/4 bars with four-bar hops; Groove uses four bars with two-bar hops. Program, velocity, tempo, instrumentation, and pitch are retained because each may participate in a style mechanism.

Codec training is per dataset: XMIDI, EMOPIA, VGMIDI, and Groove have independent experiment IDs,
optimization schedules, logs, and checkpoint directories. `pitched_union` is retained only as an
explicit legacy/ablation data group and is not used by the default reconstruction pipeline.

XMIDI latent caching excludes the same overlength rows as codec training and writes FP16
`64 x 512` posterior means. The four-GPU A100 profile uses 384 samples per GPU and 32 total loader
workers. Cache metadata includes the codec/tokenizer/manifest identities and latent shape; all
transport consumers validate it before loading a training batch.
