import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "metrabs" / "img" / "test_image_3dpw.jpg"
OUTPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "metrabs_output.npz"

MODEL_URL = "https://bit.ly/metrabs_l"


def main():
    print("Loading MeTRAbs...")
    model = hub.load(MODEL_URL)

    print(f"Image: {IMAGE_PATH}")

    image = tf.image.decode_jpeg(
        tf.io.read_file(str(IMAGE_PATH))
    )

    print("Running MeTRAbs...")
    predictions = model.detect_poses(
        image,
        skeleton="smpl+head_30",
    )

    poses3d = predictions["poses3d"].numpy()
    poses2d = predictions["poses2d"].numpy()
    boxes = predictions["boxes"].numpy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_PATH,
        poses3d=poses3d,
        poses2d=poses2d,
        boxes=boxes,
    )

    print()
    print("MeTRAbs inference complete.")
    print("poses3d:", poses3d.shape)
    print("poses2d:", poses2d.shape)
    print("boxes:", boxes.shape)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()