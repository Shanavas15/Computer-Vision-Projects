"""Count raised fingers in webcam frames with MediaPipe Hands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import cv2
import mediapipe as mp

FINGER_TIP_INDICES = (8, 12, 16, 20)


def count_raised_fingers(landmarks: Sequence, handedness: str) -> int:
    """Count raised fingers for one detected hand."""
    fingers_up = []
    if handedness == "Right":
        fingers_up.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers_up.append(landmarks[4].x > landmarks[3].x)

    fingers_up.extend(
        landmarks[tip_index].y < landmarks[tip_index - 2].y
        for tip_index in FINGER_TIP_INDICES
    )
    return sum(fingers_up)


def run_webcam(camera_index: int = 0, max_hands: int = 2) -> None:
    hands_module = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    print("Press 'q' to quit.")
    try:
        with hands_module.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam")

                frame = cv2.flip(frame, 1)
                frame_height = frame.shape[0]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                total_fingers = 0

                if results.multi_hand_landmarks and results.multi_handedness:
                    detected_hands = zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                        strict=True,
                    )
                    for index, (hand_landmarks, handedness) in enumerate(
                        detected_hands,
                    ):
                        drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hands_module.HAND_CONNECTIONS,
                        )
                        label = handedness.classification[0].label
                        count = count_raised_fingers(hand_landmarks.landmark, label)
                        total_fingers += count
                        cv2.putText(
                            frame,
                            f"{label} hand: {count}",
                            (10, 60 + index * 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )

                cv2.putText(
                    frame,
                    f"Total fingers: {total_fingers}",
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Finger Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=2)
    args = parser.parse_args()
    run_webcam(args.camera_index, args.max_hands)


if __name__ == "__main__":
    main()

