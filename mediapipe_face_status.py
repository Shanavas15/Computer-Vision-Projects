"""Estimate eye and mouth state from webcam face landmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mtcnn import MTCNN

Point = tuple[int, int]

EYE_ASPECT_RATIO_THRESHOLD = 0.23
MOUTH_ASPECT_RATIO_THRESHOLD = 0.6
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (78, 81, 13, 311, 308, 402, 14, 178, 87, 317, 324)


def euclidean_distance(first: Point, second: Point) -> float:
    return float(np.linalg.norm(np.array(first) - np.array(second)))


def ratio(vertical_distances: Sequence[float], horizontal_distance: float) -> float:
    """Calculate an aspect ratio without dividing by zero."""
    if horizontal_distance == 0:
        return 0.0
    return sum(vertical_distances) / (len(vertical_distances) * horizontal_distance)


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    return ratio(
        (
            euclidean_distance(points[1], points[5]),
            euclidean_distance(points[2], points[4]),
        ),
        euclidean_distance(points[0], points[3]),
    )


def mouth_aspect_ratio(points: Sequence[Point]) -> float:
    return ratio(
        (
            euclidean_distance(points[2], points[8]),
            euclidean_distance(points[4], points[6]),
        ),
        euclidean_distance(points[0], points[10]),
    )


def clip_box(
    box: Sequence[int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clip an MTCNN box to valid frame coordinates."""
    x, y, width, height = box
    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + width)
    bottom = min(frame_height, y + height)
    return left, top, right, bottom


def run_webcam(camera_index: int = 0) -> None:
    detector = MTCNN()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    face_mesh = mp.solutions.face_mesh
    try:
        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
        ) as mesh:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam")

                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                for detection in detector.detect_faces(rgb_frame):
                    left, top, right, bottom = clip_box(
                        detection["box"],
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if left >= right or top >= bottom:
                        continue

                    face_rgb = rgb_frame[top:bottom, left:right]
                    result = mesh.process(face_rgb)
                    if not result.multi_face_landmarks:
                        continue

                    height, width = face_rgb.shape[:2]
                    landmarks = [
                        (int(point.x * width), int(point.y * height))
                        for point in result.multi_face_landmarks[0].landmark
                    ]

                    left_eye = [landmarks[index] for index in LEFT_EYE_INDICES]
                    right_eye = [landmarks[index] for index in RIGHT_EYE_INDICES]
                    mouth = [landmarks[index] for index in MOUTH_INDICES]

                    average_eye_ratio = (
                        eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)
                    ) / 2
                    mouth_ratio = mouth_aspect_ratio(mouth)

                    eye_state = (
                        "Closed"
                        if average_eye_ratio < EYE_ASPECT_RATIO_THRESHOLD
                        else "Open"
                    )
                    mouth_state = (
                        "Open"
                        if mouth_ratio > MOUTH_ASPECT_RATIO_THRESHOLD
                        else "Closed"
                    )

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Eyes: {eye_state}",
                        (left, max(top - 30, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Mouth: {mouth_state}",
                        (left, max(top - 10, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                cv2.imshow("Face Status", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
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

