# Particle Track Reconstruction with Deep Learning
### GSoC 2026 — CMS End-to-End Deep Learning | ML4SCI Evaluation Tasks 2a & 2b

---

## Overview

This repository contains implementations of two deep learning approaches for particle track reconstruction, developed as part of the GSoC 2026 evaluation for the **CMS End-to-End (E2E) Deep Learning** project under ML4SCI.

Particle tracks — recorded as 4-momentum vectors `(E, px, py, pz)` — are the fundamental objects of jet reconstruction at the CMS detector. The goal of these tasks is to learn compact, physically meaningful representations of track-level data without full supervision, and to evaluate reconstruction quality using physics-grounded metrics such as **invariant jet mass**.

Two complementary architectures are explored:
- **Task 2a**: A Transformer-based **Masked Autoencoder (MAE)** operating on sets of tracks
- **Task 2b**: A **Graph Neural Network (GNN)** with EdgeConv-style message passing over a k-NN particle graph

---

## Repository Structure

```
.
├── task2a_mae/
│   ├── model.py           # MAE encoder-decoder with attention + gating
│   ├── train.py           # Training loop and checkpointing
│   └── evaluate.py        # Jet mass reconstruction and visualization
├── task2b_gnn/
│   ├── model.py           # EdgeConv GNN with k-NN graph construction
│   ├── train.py           # Training loop
│   └── evaluate.py        # Jet mass reconstruction and visualization
├── data/
│   └── preprocess.py      # ROOT → PyTorch tensor pipeline (fixed size: 128 tracks)
├── notebooks/
│   ├── task2a_analysis.ipynb
│   └── task2b_analysis.ipynb
├── results/
│   ├── mass_distributions/ # True vs. reconstructed jet mass plots
│   └── track_visualizations/
└── requirements.txt
```

---

## Tasks Implemented

### Task 2a — Track-Level Masked Autoencoder (MAE)

Inspired by masked self-supervised learning (He et al., 2022), this approach treats each jet as an unordered set of tracks and trains an encoder-decoder to reconstruct randomly masked track features.

**Key design choices:**
- Input: padded tensor of shape `(N, 128, 4)` — up to 128 tracks per jet, each with `(E, px, py, pz)`
- Random masking of ~40% of tracks during training
- Transformer-style encoder with multi-head self-attention and a gated feed-forward network
- Decoder reconstructs masked track 4-vectors from visible context tokens

### Task 2b — Graph Neural Network with EdgeConv

This approach constructs a particle interaction graph from track features and applies EdgeConv-style message passing to aggregate local neighborhood information — mimicking the physical intuition that nearby particles in feature space likely share a common origin.

**Key design choices:**
- k-NN graph built in `(E, px, py, pz)` feature space with `k=8`
- EdgeConv layers aggregate edge features `[x_i, x_j − x_i]` at each node
- Stacked GNN layers with residual connections and batch normalisation
- Global mean pooling to produce a jet-level latent representation

---

## Methodology

### Data Pipeline

Raw data was read from **ROOT files** using `uproot` and converted to PyTorch tensors. Tracks per event were zero-padded or truncated to a fixed size of **128 tracks** to enable batched training. For the GNN, a sparse graph is constructed on-the-fly from the track tensor during the forward pass.

```
ROOT file → uproot → numpy arrays → zero-padding (128 tracks) → PyTorch tensors
                                                                       │
                                                    GNN: k-NN graph construction (k=8)
```

### Evaluation Metric: Invariant Jet Mass

Both models are evaluated by reconstructing the **invariant jet mass** from predicted track 4-vectors and comparing against the true mass distribution:

```
M² = E_total² − |p_total|²
```

This is a physics-grounded, unsupervised quality metric: a model with good physical fidelity should reproduce the true jet mass spectrum without any direct mass supervision.

---

## Results

### Task 2a — MAE

| Metric | Observation |
|---|---|
| Training loss | Converged stably; no divergence |
| Reconstruction | Predictions collapsed toward near-zero 4-vectors |
| Jet mass distribution | Poor alignment with true distribution; mass underestimated |

The MAE learned a low-variance solution by predicting near-zero values for masked tracks — a known **trivial solution** in reconstruction tasks with continuous, sparse targets and no explicit physical constraints in the loss.

### Task 2b — GNN

| Metric | Observation |
|---|---|
| Training loss | Fast convergence; stable across epochs |
| Reconstruction | Qualitatively better per-track predictions |
| Jet mass distribution | Improved alignment with true distribution vs. MAE |

The GNN benefits from a stronger **inductive bias**: EdgeConv naturally encodes local geometric structure in feature space, which aligns with the collinear and soft-collinear structure of jet constituents. This led to more physically plausible reconstructions even without explicit physics constraints.

> **Note:** Neither model achieves full physical fidelity in the current form. Results are reported honestly; the analysis below identifies the precise failure modes and paths to resolution.

---

## Key Insights

**1. Trivial solutions are the dominant failure mode in unsupervised track reconstruction.**
Without a physics-informed loss (e.g., jet mass consistency, energy-momentum conservation), the MAE finds the lowest-variance solution — near-zero predictions — which minimises MSE but carries no physical content.

**2. GNNs outperform set-based transformers on this task due to inductive bias.**
The k-NN EdgeConv architecture encodes local feature-space proximity, which reflects real physical correlations between co-linear jet constituents. A plain MAE treats all tracks as equally independent, discarding this structure.

**3. Jet mass is a sensitive but indirect diagnostic.**
Jet mass integrates over all track predictions; a model can achieve moderate mass agreement while still reconstructing individual tracks poorly. Per-track and per-constituent metrics (e.g., constituent $p_T$ spectra) are needed alongside mass.

**4. Physics constraints belong in the loss, not the architecture alone.**
The most impactful next step is adding a differentiable physics term to the loss — penalising deviations from 4-momentum conservation or known mass constraints — rather than further architecture tuning.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
# Key dependencies: torch, torch-geometric, uproot, awkward, numpy, matplotlib, scipy
```

### Data Preprocessing

```bash
python data/preprocess.py --input /path/to/data.root --output data/processed/ --max-tracks 128
```

### Task 2a — Train MAE

```bash
python task2a_mae/train.py \
    --data-dir data/processed/ \
    --epochs 100 \
    --batch-size 64 \
    --mask-ratio 0.4 \
    --embed-dim 128 \
    --num-heads 4 \
    --lr 1e-4
```

### Task 2b — Train GNN

```bash
python task2b_gnn/train.py \
    --data-dir data/processed/ \
    --epochs 100 \
    --batch-size 64 \
    --k 8 \
    --hidden-dim 128 \
    --num-layers 3 \
    --lr 1e-3
```

### Evaluate and Plot

```bash
# Task 2a
python task2a_mae/evaluate.py --checkpoint checkpoints/mae_best.pt --data-dir data/processed/

# Task 2b
python task2b_gnn/evaluate.py --checkpoint checkpoints/gnn_best.pt --data-dir data/processed/
```

Output plots (jet mass distributions, track visualisations) are saved to `results/`.

---

## Future Work

- **Physics-informed loss functions**: Add a differentiable jet mass consistency term or 4-momentum conservation penalty directly to the training objective to prevent trivial solutions
- **Conditional generation**: Condition the MAE decoder on jet-level observables (e.g., jet $p_T$, number of constituents) to provide global context for masked track reconstruction
- **Dynamic graph construction**: Replace the static k-NN graph with a learned or dynamically updated graph (e.g., DGCNN) that adapts during training
- **Normalising flows as decoder**: Replace the MSE-trained decoder with a conditional normalising flow to model the multi-modal distribution of track 4-vectors
- **Integration with CMS E2E pipeline**: Feed learned track representations into downstream jet-level classifiers to evaluate end-to-end discrimination performance

---

## References

- He, K. et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022.
- Wang, Y. et al. (2019). *Dynamic Graph CNN for Learning on Point Clouds*. ACM TOG.
- Qu, H. & Gouskos, L. (2020). *ParticleNet: Jet Tagging via Particle Clouds*. Physical Review D.
- CMS Collaboration. *End-to-End Jet Classification at CMS*. CMS Physics Analysis Summary.
- ML4SCI GSoC 2026: [https://ml4sci.org](https://ml4sci.org)

---

## Acknowledgements

This work was developed as part of the GSoC 2026 evaluation process for the ML4SCI organisation. All mentor communication is conducted via **ml4-sci@cern.ch** in accordance with ML4SCI community guidelines.
