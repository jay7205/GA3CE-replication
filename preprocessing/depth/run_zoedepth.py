import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZOEDEPTH_ROOT = PROJECT_ROOT / "zoedepth"

if str(ZOEDEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(ZOEDEPTH_ROOT))


def estimate_depth(image_path: Union[str, Path]) -> np.ndarray:
    """
    Estimate a depth map using the official ZoeDepth repository.

    This function must run in the isolated 'zoedepth' environment.

    Parameters
    ----------
    image_path : str or Path
        Path to an RGB image.

    Returns
    -------
    np.ndarray
        Depth map with shape (H, W).

    Note
    ----
    The selected ZoeDepth checkpoint has been validated locally.
    The GA3CE supplementary material does not specify the exact
    checkpoint used in the original experiments.
    """
    import torch
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = get_config("zoedepth", "infer")
    model = build_model(config).to(device).eval()

    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        depth = model.infer_pil(image)

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(
            f"Expected a 2D depth map, got shape {depth.shape}"
        )

    if not np.isfinite(depth).all():
        raise ValueError("Depth map contains NaN or infinite values.")

    if np.any(depth <= 0):
        raise ValueError("Depth map contains non-positive values.")

    return depth


def main():
    image_path = PROJECT_ROOT / "metrabs" / "img" / "test_image_3dpw.jpg"

    output_path = (
        PROJECT_ROOT / "data" / "intermediate" / "depth_output.npz"
    )

    print("Input image:", image_path)

    depth = estimate_depth(image_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, depth=depth)

    print("Depth shape:", depth.shape)
    print("Depth dtype:", depth.dtype)
    print("Depth range:", float(depth.min()), "to", float(depth.max()))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()