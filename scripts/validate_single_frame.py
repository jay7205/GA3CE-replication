import sys
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "metrabs" / "img" / "test_image_3dpw.jpg"

POSE_PATH = ROOT / "data" / "intermediate" / "metrabs_output.npz"
DEPTH_PATH = ROOT / "data" / "intermediate" / "depth_output.npz"
MASKS_PATH = ROOT / "data" / "intermediate" / "object_masks.npz"
CONTEXT_PATH = ROOT / "data" / "intermediate" / "object_context_3d.npz"

OUTPUT_PATH = ROOT / "data" / "intermediate" / "single_frame_inputs.npz"

sys.path.insert(0, str(ROOT))

from preprocessing.pose.pose_normalization import normalize_pose


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_finite(name, array):
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def main():
    print("=" * 60)
    print("GA3CE SINGLE-FRAME INTEGRATION VALIDATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load the original image
    # --------------------------------------------------

    for path in (
        IMAGE_PATH,
        POSE_PATH,
        DEPTH_PATH,
        MASKS_PATH,
        CONTEXT_PATH,
    ):
        require_file(path)

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_width, image_height = image.size

    print("\n[1] Image")
    print("Dimensions:", image_width, "x", image_height)

    # --------------------------------------------------
    # 2. Load human pose predictions
    # --------------------------------------------------

    with np.load(POSE_PATH) as data:
        poses3d = data["poses3d"].astype(np.float32)
        poses2d = data["poses2d"].astype(np.float32)
        boxes = data["boxes"].astype(np.float32)

    if poses3d.ndim != 3 or poses3d.shape[1:] != (30, 3):
        raise ValueError(f"Unexpected 3D pose shape: {poses3d.shape}")

    if poses2d.shape != poses3d.shape[:2] + (2,):
        raise ValueError(f"Unexpected 2D pose shape: {poses2d.shape}")

    if boxes.shape != (len(poses3d), 5):
        raise ValueError(f"Unexpected box shape: {boxes.shape}")

    require_finite("3D poses", poses3d)
    require_finite("2D poses", poses2d)
    require_finite("Bounding boxes", boxes)

    print("\n[2] Human pose")
    print("People detected:", len(poses3d))
    print("3D pose shape:", poses3d.shape)
    print("2D pose shape:", poses2d.shape)
    print("Bounding-box shape:", boxes.shape)

    # MeTRAbs box format: left, top, width, height, confidence.
    box_right = boxes[:, 0] + boxes[:, 2]
    box_bottom = boxes[:, 1] + boxes[:, 3]

    if np.any(boxes[:, 2:4] <= 0):
        raise ValueError("A detected bounding box has non-positive size")

    print("Bounding boxes extend outside image:",
          bool(
              np.any(boxes[:, 0] < 0)
              or np.any(boxes[:, 1] < 0)
              or np.any(box_right > image_width)
              or np.any(box_bottom > image_height)
          ))

    # --------------------------------------------------
    # 3. Normalize the human poses
    # --------------------------------------------------

    normalized_poses = []
    t_poses = []
    scales = []

    for person_id, pose in enumerate(poses3d):
        normalized, t_pose, scale = normalize_pose(pose)

        if normalized.shape != (15, 3):
            raise ValueError(
                f"Person {person_id}: unexpected normalized shape "
                f"{normalized.shape}"
            )

        require_finite(
            f"Normalized pose for person {person_id}",
            normalized,
        )

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Person {person_id}: invalid normalization scale"
            )

        normalized_poses.append(normalized)
        t_poses.append(t_pose)
        scales.append(scale)

    normalized_poses = np.asarray(normalized_poses, dtype=np.float32)
    t_poses = np.asarray(t_poses, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)

    print("\n[3] Pose normalization")
    print("Normalized poses:", normalized_poses.shape)
    print("Translation vectors:", t_poses.shape)
    print("Scales:", scales.shape)

    # --------------------------------------------------
    # 4. Load the depth map
    # --------------------------------------------------

    with np.load(DEPTH_PATH) as data:
        depth = data["depth"].astype(np.float32)

    if depth.shape != (image_height, image_width):
        raise ValueError(
            f"Depth/image dimensions differ: "
            f"{depth.shape} vs {(image_height, image_width)}"
        )

    require_finite("Depth map", depth)

    if np.any(depth <= 0):
        raise ValueError("Depth map contains non-positive values")

    print("\n[4] Depth")
    print("Depth shape:", depth.shape)
    print("Minimum estimate:", float(depth.min()))
    print("Maximum estimate:", float(depth.max()))

    # --------------------------------------------------
    # 5. Load segmentation masks
    # --------------------------------------------------

    with np.load(MASKS_PATH) as data:
        masks = data["masks"].astype(bool)
        mask_boxes = data["boxes_xywh"].astype(np.float32)
        areas = data["areas"].astype(np.int32)

    if masks.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")

    if masks.shape[1:] != depth.shape:
        raise ValueError(
            f"Mask/depth dimensions differ: "
            f"{masks.shape[1:]} vs {depth.shape}"
        )

    if len(mask_boxes) != len(masks) or len(areas) != len(masks):
        raise ValueError("Segmentation metadata lengths differ")

    require_finite("Segmentation boxes", mask_boxes)

    print("\n[5] Segmentation")
    print("Number of regions:", len(masks))
    print("Mask tensor shape:", masks.shape)
    print("Region-box shape:", mask_boxes.shape)

    # --------------------------------------------------
    # 6. Load the region-level context
    # --------------------------------------------------

    with np.load(CONTEXT_PATH) as data:
        region_ids = data["region_ids"].astype(np.int32)
        pixels_xy = data["pixels_xy"].astype(np.float32)
        median_depths = data["median_depths"].astype(np.float32)
        valid_pixels = data["valid_pixels"].astype(np.int32)
        context_boxes = data["boxes_xywh"].astype(np.float32)
        context_areas = data["areas"].astype(np.int32)

    number_of_regions = len(region_ids)

    if pixels_xy.shape != (number_of_regions, 2):
        raise ValueError("Invalid representative-pixel array")

    if median_depths.shape != (number_of_regions,):
        raise ValueError("Invalid median-depth array")

    if valid_pixels.shape != (number_of_regions,):
        raise ValueError("Invalid valid-pixel array")

    if context_boxes.shape != (number_of_regions, 4):
        raise ValueError("Invalid context bounding-box array")

    if context_areas.shape != (number_of_regions,):
        raise ValueError("Invalid context area array")

    if np.any(region_ids < 0) or np.any(region_ids >= len(masks)):
        raise ValueError("Context contains an invalid region ID")

    require_finite("Representative pixels", pixels_xy)
    require_finite("Region median depths", median_depths)

    if np.any(median_depths <= 0):
        raise ValueError("Context contains non-positive depth")

    if np.any(pixels_xy[:, 0] < 0) or np.any(
        pixels_xy[:, 0] >= image_width
    ):
        raise ValueError("Representative X coordinate outside image")

    if np.any(pixels_xy[:, 1] < 0) or np.any(
        pixels_xy[:, 1] >= image_height
    ):
        raise ValueError("Representative Y coordinate outside image")

    print("\n[6] Object context")
    print("Valid regions:", number_of_regions)
    print("Representative pixels:", pixels_xy.shape)
    print("Median depths:", median_depths.shape)

    # --------------------------------------------------
    # 7. Save validated intermediate inputs
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        normalized_poses=normalized_poses,
        poses2d=poses2d,
        boxes=boxes,
        t_poses=t_poses,
        scales=scales,
        region_ids=region_ids,
        pixels_xy=pixels_xy,
        median_depths=median_depths,
        valid_pixels=valid_pixels,
        context_boxes_xywh=context_boxes,
        context_areas=context_areas,
    )

    print("\n[7] Integration result")
    print("Human inputs prepared:", normalized_poses.shape)
    print("Object context prepared:", number_of_regions)
    print("Saved:", OUTPUT_PATH)

    print("\nIMPORTANT:")
    print("- Depth values are monocular model estimates.")
    print("- Camera intrinsics are not yet established.")
    print("- Object positions are not calibrated 3D coordinates.")
    print("- Positional-encoding frequencies remain unverified.")
    print("- This file does not yet contain transformer-ready features.")
    print("\nPASS: Single-frame data validation completed.")


if __name__ == "__main__":
    main()