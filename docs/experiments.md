# Experiments

P0 contains the independent E00 XMIDI, E01 EMOPIA, E02 VGMIDI, and E03 Groove codec runs,
followed by E10, E11, E20, E22, E30, E31, E32, and E33. P1 contains E23, E40, E50, E51,
and E60. Config composition selects the dataset, transport, inversion, independence objective, and
optional negative control. Independent per-style flows and shuffled labels are ambiguity/causal
negative controls, never default models.

All pitched transport experiments use the BEAT VAE's `64 x 512` posterior mean. Cache metadata,
Stage-1 checkpoints, Stage-2 checkpoints, and generation are bound by exact provenance checks, so
legacy `32 x 256` artifacts cannot be mixed into a new run. E33 explicitly retains Groove's
independent `32 x 256` drum latent.

XMIDI, EMOPIA, and VGMIDI codec training and checkpoints remain completely separate. XMIDI
factorial evaluation changes genre while holding emotion, changes emotion while holding genre, and
optionally changes both. EMOPIA and VGMIDI remain separate style namespaces unless E60 explicitly
builds the joint Q1–Q4 model with dataset embeddings.
