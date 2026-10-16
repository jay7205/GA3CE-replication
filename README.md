# GA3CE Replication

Experimental implementation of GA3CE: Unconstrained 3D Gaze Estimation
with Gaze-Aware 3D Context Encoding (CVPR 2025).

## Project Overview

This project explores 3D gaze estimation using human pose information,
monocular depth estimation, object segmentation, and gaze-aware 3D context.

The goal is to build a pipeline that processes visual input and ultimately
estimates a person's 3D gaze direction.

## Current Pipeline

1. Human pose estimation using MeTRAbs.
2. Monocular depth estimation using ZoeDepth.
3. Object-region segmentation using MobileSAM.
4. Pose normalization and preliminary object-context extraction.
5. D³ positional encoding implementation.
6. Gaze-biased transformer implementation.
7. Unit tests for core components.

## Project Structure

- `ga3ce/` — Positional encoding and transformer components.
- `preprocessing/` — Pose, depth, and 3D-context utilities.
- `scripts/` — Pipeline and validation scripts.
- `tests/` — Component tests.
- `environment/` — Conda environment specifications.
- `docs/` — Implementation notes and replication specifications.
- `paper/` — Local research papers; PDFs are not currently tracked by Git.

## Environment Setup

The project uses separate Python environments for the main pipeline,
MeTRAbs, and ZoeDepth.

Environment specifications are provided in `environment/`.

Create an environment using the relevant YAML file, for example:

    conda env create -f environment/main.yml

Additional dependency repositories and pretrained model weights must be
obtained separately. See the implementation documentation before running
the complete pipeline.

## Testing

Activate the main environment and run:

    python -m pytest tests/

Individual tests can also be executed directly.

## Current Status

The human-pose, depth-estimation, segmentation, and single-frame
integration stages have been executed and validated on a test image.

Core positional-encoding and transformer components pass their current
unit tests using synthetic inputs.

The complete GA3CE pipeline is still under development. Camera geometry,
exact paper-specific configuration details, trained-model integration,
and final gaze reconstruction require further verification.

**This repository is an experimental implementation, not yet a complete
or verified reproduction of the paper.**