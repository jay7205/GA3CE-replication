import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np

from preprocessing.pose.pose_normalization import (
    normalize_pose,
    compute_cyclotorsion_rotation,
)


PROJECT_OUTPUT = "data/intermediate/metrabs_output.npz"


def main():
    # Load actual output from the official MeTRAbs model.
    data = np.load(PROJECT_OUTPUT)

    poses3d = data["poses3d"]
    poses2d = data["poses2d"]
    boxes = data["boxes"]

    print("=== MeTRAbs Input ===")
    print("3D poses:", poses3d.shape)
    print("2D poses:", poses2d.shape)
    print("Boxes:", boxes.shape)

    if poses3d.ndim != 3 or poses3d.shape[1:] != (30, 3):
        raise ValueError(
            f"Expected poses3d shape (N, 30, 3), got {poses3d.shape}"
        )

    if len(poses3d) == 0:
        raise ValueError("MeTRAbs detected no people.")

    # Process each detected person independently.
    for person_index, pose in enumerate(poses3d):
        normalized_pose, t_pose, scale = normalize_pose(pose)

        print(f"\n=== Person {person_index} ===")
        print("Normalized pose:", normalized_pose.shape)
        print("t_pose:", t_pose)
        print("Scale:", scale)

        assert normalized_pose.shape == (15, 3)
        assert t_pose.shape == (3,)
        assert np.isfinite(normalized_pose).all()
        assert np.isfinite(t_pose).all()
        assert np.isfinite(scale) and scale > 0

    # Validate the rotation function using several directions.
    test_directions = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    print("\n=== Cyclotorsion Validation ===")

    for direction in test_directions:
        unit_direction = direction / np.linalg.norm(direction)
        rotation = compute_cyclotorsion_rotation(direction)
        rotated = rotation @ unit_direction

        print("Input:", unit_direction)
        print("Rotated:", rotated)

        assert rotation.shape == (3, 3)
        assert np.allclose(
            rotated,
            np.array([0.0, 0.0, 1.0]),
            atol=1e-7,
        )

        assert np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-7,
        )

        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)

    print("\nPASS: MeTRAbs pose preprocessing and rotation checks.")


if __name__ == "__main__":
    main()