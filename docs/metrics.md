# Metrics

Codec metrics include token and note/event reconstruction plus grammar validity. Transport metrics include latent factual round trip, generated-noise recovery, NFE, and runtime. Leakage uses newly trained logistic, MLP, and temporal-CNN probes, HSIC, classwise moment deviations, pairwise tests, and projected sliced Wasserstein distance.

Style success is reported by both token and descriptor evaluators. Content descriptors remain separate: melody/chroma, onset/rhythm, density, programs, tempo, and velocity are not collapsed into a single causal content score. Identity, cycle, exogenous consistency, composition, factorial commutativity, and cross-seed spread are evaluated without paired targets.
