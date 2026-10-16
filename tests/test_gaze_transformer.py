import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ga3ce.transformer.gaze_transformer import GazeTransformer


def main():
    torch.manual_seed(42)

    print("Starting transformer tests...")

    model = GazeTransformer()
    model.eval()

    batch_size = 2
    num_objects = 12

    object_features = torch.randn(batch_size, num_objects, 256)
    subject_features = torch.randn(batch_size, 362)
    view_directions = torch.randn(batch_size, 3)
    object_positions = torch.randn(batch_size, num_objects, 3)

    padding_mask = torch.zeros(
        batch_size, num_objects, dtype=torch.bool
    )
    padding_mask[0, -2:] = True

    # Test forward pass.
    with torch.no_grad():
        output = model(
            object_features=object_features,
            subject_features=subject_features,
            view_directions=view_directions,
            object_positions=object_positions,
            object_padding_mask=padding_mask,
        )

    assert output["residual_gaze"].shape == (2, 3)
    assert output["encoded_objects"].shape == (2, 12, 256)
    assert output["attention_bias"].shape == (2, 1, 12)

    assert torch.isfinite(output["residual_gaze"]).all()
    assert torch.isfinite(output["encoded_objects"]).all()
    assert torch.isfinite(output["attention_bias"]).all()

    print("PASS: Output shapes")
    print("PASS: Finite values")
    print("PASS: Padding mask")

    # Test gradient flow.
    model.train()
    model.zero_grad(set_to_none=True)

    output = model(
        object_features=object_features,
        subject_features=subject_features,
        view_directions=view_directions,
        object_positions=object_positions,
        object_padding_mask=padding_mask,
    )

    loss = output["residual_gaze"].square().mean()
    loss.backward()

    gradient = model.residual_gaze_head[0].weight.grad

    assert gradient is not None
    assert torch.isfinite(gradient).all()

    print("PASS: Gradient flow")
    print("Residual gaze shape:", tuple(output["residual_gaze"].shape))
    print("ALL TESTS PASSED")
    print("Note: Synthetic inputs; this is not a trained gaze estimator.")


if __name__ == "__main__":
    main()