# Forehead Teleport

A real-time, gesture-controlled teleportation visual effect built with Python, OpenCV, and MediaPipe.

## Features

- Real-time webcam input
- Hand and face landmark detection
- Finger-to-forehead gesture recognition
- Person segmentation
- Particle-based dust teleportation effect
- OBS Virtual Camera support

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## Controls

| Key | Action |
|---|---|
| `B` | Capture the empty-room background |
| `Q` | Quit the application |

## Installation

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required MediaPipe model files and place them in the `models/` folder. See the model setup instructions in this repository.

## Run

```bash
python app.py
```

## OBS Integration

The webcam effect can be used in streaming or video-call applications through OBS Virtual Camera.

## License

License to be decided.