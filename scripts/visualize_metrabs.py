from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "metrabs" / "img" / "test_image_3dpw.jpg"
POSE_PATH = ROOT / "data" / "intermediate" / "metrabs_output.npz"
OUTPUT_PATH = ROOT / "outputs" / "metrabs_visualization.jpg"


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)

    data = np.load(POSE_PATH)
    keypoints = data["poses2d"]
    boxes = data["boxes"]

    width, height = image.size

    for person_id, (points, box) in enumerate(zip(keypoints, boxes)):
        left, top, box_width, box_height, confidence = box

        # Clip the bounding box to the image boundaries.
        x1 = int(np.clip(left, 0, width - 1))
        y1 = int(np.clip(top, 0, height - 1))
        x2 = int(np.clip(left + box_width, 0, width - 1))
        y2 = int(np.clip(top + box_height, 0, height - 1))

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text(
            (x1, max(0, y1 - 20)),
            f"Person {person_id}: {confidence:.3f}",
            fill="red",
        )

        for joint_id, (x, y) in enumerate(points):
            if not np.isfinite([x, y]).all():
                continue

            if 0 <= x < width and 0 <= y < height:
                radius = 4
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill="lime",
                )
                draw.text((x + 4, y), str(joint_id), fill="yellow")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)

    print("Saved visualization:", OUTPUT_PATH)


if __name__ == "__main__":
    main()