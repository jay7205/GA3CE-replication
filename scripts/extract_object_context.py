from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MASKS_PATH = ROOT / "data" / "intermediate" / "object_masks.npz"
DEPTH_PATH = ROOT / "data" / "intermediate" / "depth_output.npz"
OUTPUT_PATH = ROOT / "data" / "intermediate" / "object_context.npz"


def main():
    if not MASKS_PATH.is_file():
        raise FileNotFoundError(MASKS_PATH)

    if not DEPTH_PATH.is_file():
        raise FileNotFoundError(DEPTH_PATH)

    with np.load(MASKS_PATH) as data:
        masks = data["masks"].astype(bool)
        boxes = data["boxes_xywh"].astype(np.float32)
        areas = data["areas"].astype(np.int32)

    with np.load(DEPTH_PATH) as data:
        depth = data["depth"].astype(np.float32)

    if masks.ndim != 3:
        raise ValueError(f"Expected masks (N,H,W), got {masks.shape}")

    if depth.ndim != 2:
        raise ValueError(f"Expected depth (H,W), got {depth.shape}")

    if masks.shape[1:] != depth.shape:
        raise ValueError(
            f"Mask/depth mismatch: {masks.shape[1:]} vs {depth.shape}"
        )

    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Depth contains invalid values")

    n = len(masks)

    median_depth = np.full(n, np.nan, dtype=np.float32)
    mean_depth = np.full(n, np.nan, dtype=np.float32)
    depth_std = np.full(n, np.nan, dtype=np.float32)
    valid_pixels = np.zeros(n, dtype=np.int32)

    for i, mask in enumerate(masks):
        values = depth[mask]
        values = values[np.isfinite(values) & (values > 0)]

        valid_pixels[i] = len(values)

        if len(values) == 0:
            continue

        median_depth[i] = np.median(values)
        mean_depth[i] = np.mean(values)
        depth_std[i] = np.std(values)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        boxes_xywh=boxes,
        areas=areas,
        median_depth=median_depth,
        mean_depth=mean_depth,
        depth_std=depth_std,
        valid_pixels=valid_pixels,
    )

    print("Object-context extraction complete.")
    print("Number of regions:", n)
    print("Depth map shape:", depth.shape)
    print("Saved:", OUTPUT_PATH)

    print("\nLargest regions:")
    for i in np.argsort(areas)[::-1][:10]:
        print(
            f"Region {i:02d} | "
            f"area={areas[i]:7d} | "
            f"median_depth={median_depth[i]:.4f} | "
            f"depth_std={depth_std[i]:.4f} | "
            f"valid_pixels={valid_pixels[i]}"
        )

    print(
        "\nNote: Depth values are monocular model estimates. "
        "They are not calibrated metric distances."
    )


if __name__ == "__main__":
    main()