from typing import Optional

import numpy as np


def validate_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """Validate a 3x3 pinhole camera intrinsic matrix."""
    K = np.asarray(intrinsics, dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError(
            f"Expected intrinsics shape (3, 3), got {K.shape}"
        )

    if not np.isfinite(K).all():
        raise ValueError("Intrinsics contain non-finite values")

    if abs(K[0, 0]) < 1e-12 or abs(K[1, 1]) < 1e-12:
        raise ValueError("Focal lengths must be nonzero")

    if not np.allclose(K[2], [0.0, 0.0, 1.0]):
        raise ValueError(
            "Expected a conventional pinhole intrinsic matrix"
        )

    return K


def backproject_pixel(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Backproject one pixel using a pinhole camera model.

    Assumptions:
      - pixel coordinates are (x, y) = (column, row)
      - depth is camera-axis Z depth
      - intrinsics correspond to the same image resolution
      - output is in the depth map's distance units

    These assumptions must be verified for the selected models.
    """
    K = validate_intrinsics(intrinsics)

    values = np.asarray(
        [pixel_x, pixel_y, depth],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError("Pixel coordinates and depth must be finite")

    if depth <= 0:
        raise ValueError("Depth must be positive")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (pixel_x - cx) * depth / fx
    Y = (pixel_y - cy) * depth / fy
    Z = depth

    return np.asarray([X, Y, Z], dtype=np.float32)


def backproject_pixels(
    pixels_xy: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Backproject N image pixels into N camera-coordinate 3D points."""
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    depth_values = np.asarray(depths, dtype=np.float64)

    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(
            f"Expected pixels shape (N, 2), got {pixels.shape}"
        )

    if depth_values.shape != (len(pixels),):
        raise ValueError(
            f"Expected depths shape ({len(pixels)},), "
            f"got {depth_values.shape}"
        )

    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.stack([
        backproject_pixel(x, y, z, intrinsics)
        for (x, y), z in zip(pixels, depth_values)
    ])


def representative_mask_point(
    mask: np.ndarray,
    depth_map: np.ndarray,
    min_valid_depth: Optional[float] = None,
    max_valid_depth: Optional[float] = None,
) -> tuple:
    """
    Return a mask's median image location and median valid depth.

    Returns:
        pixel_xy: [x, y]
        median_depth: scalar
        valid_pixel_count: number of accepted pixels
    """
    mask = np.asarray(mask, dtype=bool)
    depth_map = np.asarray(depth_map, dtype=np.float32)

    if mask.ndim != 2 or depth_map.ndim != 2:
        raise ValueError("Mask and depth map must both be 2D")

    if mask.shape != depth_map.shape:
        raise ValueError(
            f"Mask/depth shape mismatch: {mask.shape} vs {depth_map.shape}"
        )

    valid = mask & np.isfinite(depth_map) & (depth_map > 0)

    if min_valid_depth is not None:
        valid &= depth_map >= min_valid_depth

    if max_valid_depth is not None:
        valid &= depth_map <= max_valid_depth

    ys, xs = np.where(valid)

    if len(xs) == 0:
        raise ValueError("Mask has no valid depth pixels")

    pixel_xy = np.asarray(
        [np.median(xs), np.median(ys)],
        dtype=np.float32,
    )

    median_depth = float(np.median(depth_map[valid]))

    return pixel_xy, median_depth, len(xs)