import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.backprojection.depth_backprojection import (
    backproject_pixel,
    representative_mask_point,
)


MASKS_PATH = ROOT / "data" / "intermediate" / "object_masks.npz"
DEPTH_PATH = ROOT / "data" / "intermediate" / "depth_output.npz"
OUTPUT_PATH = ROOT / "data" / "intermediate" / "object_context_3d.npz"

# Leave as None until the correct camera intrinsics are established.
CAMERA_INTRINSICS = None


def main():
    with np.load(MASKS_PATH) as data:
        masks = data["masks"].astype(bool)
        boxes = data["boxes_xywh"].astype(np.float32)
        areas = data["areas"].astype(np.int32)

    with np.load(DEPTH_PATH) as data:
        depth_map = data["depth"].astype(np.float32)

    if masks.shape[1:] != depth_map.shape:
        raise ValueError(
            f"Mask/depth mismatch: {masks.shape[1:]} vs {depth_map.shape}"
        )

    records = []

    for region_id, mask in enumerate(masks):
        try:
            pixel_xy, median_depth, valid_count = representative_mask_point(
                mask,
                depth_map,
            )
        except ValueError:
            print(f"Skipping region {region_id}: no valid depth")
            continue

        record = {
            "region_id": region_id,
            "pixel_xy": pixel_xy,
            "median_depth": median_depth,
            "valid_pixels": valid_count,
        }

        if CAMERA_INTRINSICS is not None:
            record["position_3d"] = backproject_pixel(
                float(pixel_xy[0]),
                float(pixel_xy[1]),
                median_depth,
                CAMERA_INTRINSICS,
            )

        records.append(record)

    if not records:
        raise RuntimeError("No valid object regions were found")

    region_ids = np.asarray(
        [r["region_id"] for r in records],
        dtype=np.int32,
    )
    pixels_xy = np.asarray(
        [r["pixel_xy"] for r in records],
        dtype=np.float32,
    )
    median_depths = np.asarray(
        [r["median_depth"] for r in records],
        dtype=np.float32,
    )
    valid_pixels = np.asarray(
        [r["valid_pixels"] for r in records],
        dtype=np.int32,
    )

    output = {
        "region_ids": region_ids,
        "pixels_xy": pixels_xy,
        "median_depths": median_depths,
        "valid_pixels": valid_pixels,
        "boxes_xywh": boxes[region_ids],
        "areas": areas[region_ids],
    }

    if CAMERA_INTRINSICS is not None:
        output["positions_3d"] = np.asarray(
            [r["position_3d"] for r in records],
            dtype=np.float32,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **output)

    print("3D context preparation complete.")
    print("Valid regions:", len(records))
    print("Representative pixels:", pixels_xy.shape)
    print("Median depths:", median_depths.shape)
    print("3D positions calculated:", CAMERA_INTRINSICS is not None)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()