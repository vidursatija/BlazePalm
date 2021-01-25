# BlazePalm

BlazeFace is a fast, light-weight 2-part hand landmark detector from Google Research. [Read more](https://google.github.io/mediapipe/solutions/hands.html), [Paper on arXiv](https://arxiv.org/abs/2006.10214)

A pretrained model is available as part of Google's [MediaPipe](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md) framework.

![](https://google.github.io/mediapipe/images/mobile/hand_tracking_3d_android_gpu.gif)

Besides a bounding box, BlazePalm also predicts 21 3D keypoints for hand landmarks (5 fingers x 4 keypoints + 1 wrist)

Because BlazeFace is designed for use on mobile devices, the pretrained model is in TFLite format. However, I wanted to use it in **PyTorch** and not *TensorFlow*. I also ported the PyTorch model to **CoreML** because the model is made for phones.

There are 2 parts to the model:
1. Hand detector. It is a classic single shot detector(SSD).
2. Landmark detector. After getting the hands from the hand detector, we crop them and pass them through the landmark detector to get the 3D landmarks.

## Inside this repo

Essential ML files:

- **ML/blazepalm.py**: defines the `PalmDetector` class that finds the bounding box for the hands in an image.

- **ML/palmdetector.pth**: the weights for the trained model for `PalmDetector`

- **ML/genanchors.py**: creates anchor boxes and saves them as a binary file (ML/anchors.npy)

- **ML/anchors.npy**: lookup table with anchor boxes for `PalmDetector`

- **ML/export_detector.py**: For converting the PyTorch model of  `PalmDetector` to CoreML.

- **ML/handlandmarks.py**: defines the `HandLandmarks` class that finds the 3D landmarks for a hand.

- **ML/HandLandmarks.pth**: the weights for the trained model for `HandLandmarks`

- **ML/export_landmarks.py**: For converting the PyTorch model of `HandLandmarks` to CoreML.

ML Notebooks:

- **ML/ConvertPalmDetector.ipynb**: loads the weights from the TFLite model of `PalmDetector` and converts them to PyTorch format (ML/palmdetector.pth)

- **ML/ConvertHandDetector.ipynb**: loads the weights from the TFLite model of `HandLandmarks` and converts them to PyTorch format (ML/HandLandmarks.pth)

iOS CoreML App

- **App/**

## Detections

### CoreML

Each hand detection has 2 SIMD2 vectors(bounding box) and 1 Float number(confidence):

- The first 4 numbers describe the bounding box corners: 
    - `xmin, ymin, xmax, ymax`

- These are normalized coordinates (between 0 and 1).

- SIMD2 is used for faster vector math.

- The final number is the confidence score that this detection really is a hand.


Each landmark detection has 21 SIMD3 vectors(landmarks) and 1 Float number(confidence):
![](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

- These are normalized coordinates (between 0 and 1).

- SIMD3 is used for faster vector math.

- The final number is the confidence score that this detection really is a hand.

### PyTorch

Each hand detection has 4 Floats(bounding box) and 1 Float number(confidence) as defined in the CoreML section.
Each landmark detection has 21 Floats(landmarks) and 1 Float number(confidence) as defined in the CoreML section.
