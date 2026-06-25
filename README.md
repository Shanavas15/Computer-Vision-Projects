# Computer Vision Projects

## Installation

Install only the dependencies needed for the demo you want to run:

```powershell
pip install -r requirements-base.txt
pip install -r requirements-mediapipe.txt
pip install -r requirements-deepface.txt
```

## Examples

```powershell
python haar_image_detection.py path\to\image.jpg --output output\annotated.jpg
python haar_webcam_detection.py
python deepface_verification.py first.jpg second.jpg --plot
python mediapipe_face_status.py
python mediapipe_finger_counter.py
```

Press `q` to stop the webcam demos. The face-status demo also accepts `Esc`.

## Notes

- Haar cascades are convenient demos, but modern detectors are usually more
  accurate for production applications.
- Face verification should be used with consent and tested carefully for the
  intended environment.


