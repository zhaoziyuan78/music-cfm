# Metrics

Codec metrics include token and note/event reconstruction plus grammar validity. Transport metrics include latent factual round trip, generated-noise recovery, NFE, and runtime. Leakage uses newly trained logistic, MLP, and temporal-CNN probes, HSIC, classwise moment deviations, pairwise tests, and projected sliced Wasserstein distance.

Style success is reported by zero-shot CLaMP 2 similarity between generated MIDI and fixed
style-text prompts. Content metrics remain separate: pitch-class histogram, melody contour,
density, programs, tempo, and velocity are not collapsed into a single causal content score.
Independent linear, MLP, and temporal probes measure style leakage from abducted noise, while
reference-free MIDI validity, duration, pitch range, pitch-class coverage, and density diagnose
generation quality. Evaluation does not require paired counterfactual targets.
