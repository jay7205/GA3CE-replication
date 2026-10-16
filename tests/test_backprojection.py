import sys
from pathlib import Path

import numpy as np

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.backprojection.depth_backprojection import (
    backproject_pixel,
    backproject_pixels,
    representative_mask_point,
)

def main():
    # Synthetic camera used only to test the mathematics.
    K = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 40.0],
        [0.0, 0.0, 1.0],
    ])

    # Principal point should backproject onto the optical axis.
    point = backproject_pixel(50, 40, 2.0, K)
    np.testing.assert_allclose(point, [0, 0, 2], atol=1e-6)

    # Off-axis pixel.
    point = backproject_pixel(60, 50, 2.0, K)
    np.testing.assert_allclose(point, [0.2, 0.2, 2], atol=1e-6)

    points = backproject_pixels(
        np.array([[50, 40], [60, 50]]),
        np.array([2.0, 2.0]),
        K,
    )

    assert points.shape == (2, 3)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    depth = np.full((4, 4), 2.0, dtype=np.float32)

    pixel_xy, median_depth, count = representative_mask_point(
        mask, depth
    )

    np.testing.assert_allclose(pixel_xy, [1.5, 1.5])
    assert median_depth == 2.0
    assert count == 4

    print("PASS: Synthetic backprojection tests.")
    print("Camera intrinsics in these tests are synthetic.")


if __name__ == "__main__":
    main()