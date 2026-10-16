import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ga3ce.positional_encoding.d3_encoding import (
    D3PositionalEncoding,
)


def main():
    # Synthetic frequencies for testing only.
    # These are NOT claimed to be the paper's frequencies.
    encoder = D3PositionalEncoding(
        direction_frequencies=[1.0],
        distance_frequencies=[1.0, 2.0],
    )

    points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = encoder(points)

    assert output["direction"].shape == (3, 3)
    assert output["distance"].shape == (3, 1)

    # One frequency: 3 coordinates * sin/cos = 6.
    assert output["direction_features"].shape == (3, 6)

    # Two frequencies: 1 distance * 2 frequencies * sin/cos = 4.
    assert output["distance_features"].shape == (3, 4)

    assert output["d3_features"].shape == (3, 10)
    assert torch.isfinite(output["d3_features"]).all()

    output["d3_features"].sum().backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()

    print("PASS: D3 positional encoding shape checks.")
    print("PASS: Finite outputs and gradient checks.")
    print("Output shape:", tuple(output["d3_features"].shape))
    print("Note: Synthetic frequencies used for testing.")


if __name__ == "__main__":
    main()