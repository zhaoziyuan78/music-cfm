# Methodology

The main method transforms the complete frozen VAE latent. I-CFM learns the conditional vector field along straight paths from standard Gaussian noise to factual latents. Abduction integrates the same field from time one to zero; prediction integrates the same grid from zero to one under the intervention label. Conditional DDIM uses deterministic sampling plus vanilla or fixed-point inversion.

Task-aware conditioning activates only the observed style slot for a genre or emotion task. A
factorial task instead uses a constant style sentinel and separate genre/emotion slots, with one
axis changed per intervention. Stage-2 fine-tuning retains the base generative loss and adds
factual round-trip loss plus global-batch HSIC, sliced-Wasserstein prior matching, and cross-class
MMD over resampled random projections, token summaries, and random latent blocks. These are
distributional exogeneity objectives, not identifiability proofs.

The optional weak split applies an invertible orthogonal feature rotation and names the subspaces “conserved” and “editable”; it does not claim that either is a true content or confounder latent.
