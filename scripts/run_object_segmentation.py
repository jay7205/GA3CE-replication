from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "metrabs" / "img" / "test_image_3dpw.jpg"
CHECKPOINT_PATH = ROOT / "MobileSAM" / "weights" / "mobile_sam.pt"

OUTPUT_DIR = ROOT / "data" / "intermediate"
VISUALIZATION_PATH = ROOT / "outputs" / "object_segmentation.jpg"
MASKS_PATH = OUTPUT_DIR / "object_masks.npz"


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading MobileSAM...")
    print("Device:", device)

    model = sam_model_registry["vit_t"](
        checkpoint=str(CHECKPOINT_PATH)
    )

    model.to(device=device)
    model.eval()

    return model, device


# --------------------------------------------------
# Segmentation
# --------------------------------------------------

def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_array = np.asarray(image)

    model, device = load_model()

    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=500,
    )

    print("Image dimensions:", image.size)
    print("Generating object-region masks...")

    with torch.inference_mode():
        masks = mask_generator.generate(image_array)

    print("Total masks generated:", len(masks))

    # Sort larger regions first for easier inspection.
    masks.sort(key=lambda item: item["area"], reverse=True)

    # Save masks and useful metadata.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if masks:
        mask_array = np.stack(
            [item["segmentation"] for item in masks]
        ).astype(bool)

        areas = np.asarray(
            [item["area"] for item in masks],
            dtype=np.int32,
        )

        bounding_boxes = np.asarray(
            [item["bbox"] for item in masks],
            dtype=np.float32,
        )

        predicted_iou = np.asarray(
            [item["predicted_iou"] for item in masks],
            dtype=np.float32,
        )

    else:
        height, width = image_array.shape[:2]

        mask_array = np.empty(
            (0, height, width),
            dtype=bool,
        )

        areas = np.empty(0, dtype=np.int32)
        bounding_boxes = np.empty((0, 4), dtype=np.float32)
        predicted_iou = np.empty(0, dtype=np.float32)

    np.savez_compressed(
        MASKS_PATH,
        masks=mask_array,
        areas=areas,
        boxes_xywh=bounding_boxes,
        predicted_iou=predicted_iou,
    )

    # Draw mask boundaries for visual inspection.
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for index, item in enumerate(masks):
        segmentation = item["segmentation"]

        # Find the boundary pixels of each binary mask.
        padded = np.pad(
            segmentation.astype(np.uint8),
            1,
            mode="constant",
        )

        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )

        boundary = segmentation & ~interior
        ys, xs = np.where(boundary)

        # Draw only the boundary, not the entire region.
        if len(xs):
            draw.point(
                list(zip(xs.tolist(), ys.tolist())),
                fill="red",
            )

        x, y, width, height = item["bbox"]

        draw.text(
            (int(x), int(y)),
            f"Region {index}",
            fill="yellow",
        )

    VISUALIZATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay.save(VISUALIZATION_PATH)

    print("\nSegmentation complete.")
    print("Masks shape:", mask_array.shape)
    print("Saved masks:", MASKS_PATH)
    print("Saved visualization:", VISUALIZATION_PATH)

    if len(masks):
        print("\nLargest regions:")
        for index, item in enumerate(masks[:10]):
            print(
                f"Region {index}: "
                f"area={item['area']}, "
                f"bbox={item['bbox']}, "
                f"IoU={item['predicted_iou']:.3f}"
            )


if __name__ == "__main__":
    main()