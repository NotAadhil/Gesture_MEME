# Gesture Meme

A computer vision project that detects face and hand gestures in real-time using a webcam and displays a corresponding meme image.

## Technologies

- **Python 3.11** (recommended compatible environment)
- **OpenCV**
- **MediaPipe**
- **NumPy**

## Requirements

- **Python 3.8 - 3.11** (MediaPipe solution API is not compatible with Python 3.12+)
- **Windows OS**

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Detected Gestures

| Gesture | Meme File |
|---------|-----------|
| **Two hands forming a 'T' (timeout gesture)** | `Timeout.png` |
| **Palms pressed together (praying gesture) in front of the face** | `son.png` |
| **Raised or furrowed eyebrows** | `dog.jpeg` |
| **Sticking tongue out** | `cat.png` |
| **Finger touching/biting the mouth (shush/bite gesture)** | `bite.png` |
| **Two hands on the sides of the face (cinema framing gesture)** | `cinema.jpg` |
| **Two hands above the nose** | `Sonic.jpeg` |
| **Index and middle fingers extended** | `rat.jpeg` |

## Project Structure

```text
gesture_meme/
├── main.py
├── requirements.txt
├── run.bat
├── cinema.jpg
├── bite.png
├── cat.png
├── dog.jpeg
├── rat.jpeg
├── Sonic.jpeg
├── son.png
└── Timeout.png
```

## How to Use (Windows)

1. Clone the repository.
2. Put the meme images in the same folder as `main.py`.
3. Double-click the `run.bat` file. It will automatically set up the virtual environment, install the correct compatible library versions, and start the application.
4. When the application starts, look straight ahead with a neutral face during **calibration**.
5. Once calibrated, try the gestures in front of the camera!
6. Click the **QUIT** button in the window, press **Esc**, or press **Q** to exit.

## Notes

- **Calibration** takes a few seconds at startup; it is required for gesture detection to calibrate thresholds correctly for your face.
- Meme images must be located in the **same directory** as `main.py`.
- Works best in **good lighting** conditions.
- Uses `CAP_DSHOW` backend for Windows webcam capture.
