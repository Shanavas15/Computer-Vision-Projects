"""Run Haar cascade detection on webcam frames."""

from __future__ import annotations

import argparse

import cv2

from haar_image_detection import annotate_image


def run_webcam(camera_index: int = 0) -> None:
    """Display annotated webcam frames until the user presses q."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    print("Press 'q' to quit.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam")

            cv2.imshow("Webcam Haar Cascade Detection", annotate_image(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    run_webcam(args.camera_index)


if __name__ == "__main__":
    main()

