"""Compare two face images with DeepFace."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def verify_faces(first_image: Path, second_image: Path) -> dict[str, Any]:
    """Return DeepFace verification details for two existing image files."""
    from deepface import DeepFace

    for image_path in (first_image, second_image):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

    return DeepFace.verify(
        img1_path=str(first_image),
        img2_path=str(second_image),
    )


def plot_comparison(
    first_image: Path,
    second_image: Path,
    *,
    verified: bool,
) -> None:
    """Display the compared images and verification result."""
    import cv2
    import matplotlib.pyplot as plt

    images = []
    for image_path in (first_image, second_image):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(
        axes,
        images,
        ("First image", "Second image"),
        strict=True,
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    match_text = "MATCH" if verified else "NO MATCH"
    plt.suptitle(match_text, fontsize=16, color="green" if verified else "red")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_image", type=Path)
    parser.add_argument("second_image", type=Path)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display the compared images after verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_faces(args.first_image, args.second_image)
    print(f"Verified: {result['verified']}")
    print(f"Distance: {result.get('distance', 'not reported')}")
    print(f"Threshold: {result.get('threshold', 'not reported')}")

    if args.plot:
        plot_comparison(
            args.first_image,
            args.second_image,
            verified=bool(result["verified"]),
        )


if __name__ == "__main__":
    main()

