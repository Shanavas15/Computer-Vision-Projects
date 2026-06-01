"""Annotate faces, eyes, smiles, and upper bodies in an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_cascade(filename: str) -> cv2.CascadeClassifier:
    """Load one of OpenCV's bundled Haar cascade classifiers."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {filename}")
    return cascade


FACE_CASCADE = load_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE = load_cascade("haarcascade_eye.xml")
SMILE_CASCADE = load_cascade("haarcascade_smile.xml")
UPPER_BODY_CASCADE = load_cascade("haarcascade_upperbody.xml")


def draw_label(
    image: np.ndarray,
    label: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.6,
) -> None:
    """Draw a readable label while keeping its baseline inside the image."""
    x, y = origin
    cv2.putText(
        image,
        label,
        (x, max(y, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def annotate_image(image: np.ndarray) -> np.ndarray:
    """Return an annotated copy of a BGR image."""
    annotated = image.copy()
    gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)

    for x, y, width, height in UPPER_BODY_CASCADE.detectMultiScale(gray, 1.1, 5):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (128, 0, 128),
            2,
        )
        draw_label(annotated, "Upper body", (x, y - 10), (128, 0, 128))

    for x, y, width, height in FACE_CASCADE.detectMultiScale(gray, 1.3, 5):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2,
        )
        draw_label(annotated, "Face", (x, y - 10), (255, 0, 0))

        face_gray = gray[y : y + height, x : x + width]
        face_color = annotated[y : y + height, x : x + width]

        for eye_x, eye_y, eye_width, eye_height in EYE_CASCADE.detectMultiScale(
            face_gray,
            1.1,
            10,
        ):
            cv2.rectangle(
                face_color,
                (eye_x, eye_y),
                (eye_x + eye_width, eye_y + eye_height),
                (0, 255, 0),
                2,
            )

        for smile_x, smile_y, smile_width, smile_height in (
            SMILE_CASCADE.detectMultiScale(face_gray, 1.7, 22)
        ):
            cv2.rectangle(
                face_color,
                (smile_x, smile_y),
                (smile_x + smile_width, smile_y + smile_height),
                (0, 255, 255),
                2,
            )

    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--output", type=Path, help="Optional annotated image path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a Matplotlib preview window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read image: {args.image}")

    annotated = annotate_image(image)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), annotated):
            raise OSError(f"Could not write image: {args.output}")
        print(f"Saved annotated image to {args.output}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Haar Cascade Detection")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

