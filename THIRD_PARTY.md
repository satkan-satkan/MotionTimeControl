# Third-party components

The MIT license in this repository covers the project code and its original procedural demo animation code. It does not replace the licenses of dependencies, downloaded models or user-supplied videos.

- [MediaPipe](https://github.com/google-ai-edge/mediapipe): hand landmark detection.
- [Official Hand Landmarker model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker): `setup_assets.py` downloads Google's model version 1 and checks its SHA-256. The model is not committed to this repository.
- [OpenCV](https://github.com/opencv/opencv) and [opencv-python packaging](https://github.com/opencv/opencv-python): camera input, video decoding, drawing, and the desktop window.
- [NumPy](https://github.com/numpy/numpy): image arrays.

## Optional real footage

The development prototype used local clips from these public source URLs:

- Flower: https://www.youtube.com/watch?v=LjCzPp-MK48 — source interval 01:16–01:23. Production/editing credited in the video's end titles to David de los Santos.
- Rocket: https://www.youtube.com/watch?v=l2o-2imrjeA — source interval 00:18–00:25.

Neither footage nor screenshots derived from it are distributed here. Permission to republish those videos has not been established by this project. Use your own footage or material you have permission to use. The generated demos need no external video downloads.
