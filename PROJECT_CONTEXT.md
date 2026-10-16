# GA3CE Replication Project

## Main Goal

Replicate the research paper:

GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding

CVPR 2025
Kawana et al.

The goal is a high-fidelity replication of the published method.

## Existing GAFA Baseline

The sibling folder:

../dynamic-3d-gaze-from-afar

is our VERIFIED GAFA baseline.

It has already been successfully run on:
- Windows
- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- RTX 4060 Laptop GPU
- GAFA dataset
- pretrained GAFA model

Verified GAFA 3D MAE:
- Front: 20.699°
- Back: 23.212°
- All: 21.690°

IMPORTANT:
The GAFA folder is the baseline/reference.
DO NOT modify it.

## GA3CE Project Rules

All new GA3CE code must stay inside this folder.

Use a separate Conda environment named:

ga3ce

Do NOT install GA3CE dependencies into the existing `gafa` environment.

Do NOT blindly copy the old GAFA environment.

## Paper

The main paper is:

GA3CE_CVPR2025.pdf

The supplementary material should also be used when available.

The paper describes a pipeline involving:

RGB + depth + camera intrinsics
→ 3D human pose
→ view direction
→ MobileSAM object masks
→ 3D object positions
→ egocentric transformation
→ cyclotorsion rotation
→ D3 positional encoding
→ Transformer
→ 3D gaze direction

Important reported settings:

Npose = 15
Clatent = 256
Cgaze = 106
Ckeypoint = 10
Transformer encoder layers = 3
Transformer decoder layers = 3
Optimizer = AdamW
Learning rate = 0.0014
Epochs = 20

## Development Rule

Do NOT start implementing immediately.

First inspect:
1. Main paper
2. Supplementary material
3. Existing GAFA baseline
4. Required pretrained models
5. Required datasets
6. Dependencies and version conflicts

Then create an implementation plan.

Prioritize fidelity to the paper over convenience.

If the paper does not specify something, identify it instead of inventing it.

The GAFA baseline must remain untouched.