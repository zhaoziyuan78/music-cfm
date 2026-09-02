# Methodology

The main method transforms the complete frozen VAE latent. I-CFM learns the conditional vector field along straight paths from standard Gaussian noise to factual latents. Abduction integrates the same field from time one to zero; prediction integrates the same grid from zero to one under the intervention label. Conditional DDIM is matched in backbone capacity and uses deterministic sampling plus vanilla or fixed-point inversion.

Stage-2 fine-tuning retains the base generative loss and adds factual round-trip loss, normalized HSIC between a fixed projection of abducted noise and style, and classwise Gaussian moment matching. These are distributional exogeneity objectives, not identifiability proofs.

The optional weak split applies an invertible orthogonal feature rotation and names the subspaces “conserved” and “editable”; it does not claim that either is a true content or confounder latent.
