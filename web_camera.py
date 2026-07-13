from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template_string


# -----------------------------------
# Settings
# -----------------------------------
MODEL_PATH = Path("models/food_model.keras")
LABELS_PATH = Path("labels.txt")

CAMERA_INDEX = 0
IMAGE_SIZE = (224, 224)
PREDICT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.60


# -----------------------------------
# Check required files
# -----------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")


# -----------------------------------
# Load model and labels
# -----------------------------------
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    line.strip()
    for line in LABELS_PATH.read_text().splitlines()
    if line.strip()
]

if len(class_names) != 2:
    raise ValueError(
        f"Expected exactly 2 labels, but found: {class_names}"
    )

print("Classes:", class_names)


# -----------------------------------
# Open camera
# -----------------------------------
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open /dev/video0. Make sure no other program is using it."
    )


# Prevent multiple browser requests from reading
# the camera simultaneously
camera_lock = Lock()

app = Flask(__name__)


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Healthy Food Detector</title>

    <style>
        body {
            margin: 0;
            min-height: 100vh;
            background: #111827;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: min(900px, 94%);
            text-align: center;
        }

        h1 {
            margin-bottom: 8px;
        }

        p {
            color: #cbd5e1;
        }

        img {
            width: 100%;
            max-width: 760px;
            border: 4px solid #334155;
            border-radius: 16px;
            background: black;
        }

        .note {
            margin-top: 15px;
            font-size: 14px;
        }
    </style>
</head>

<body>
    <main class="container">
        <h1>Healthy Food Detector</h1>
        <p>Jetson Orin Nano live camera classification</p>

        <img
            src="/video_feed"
            alt="Live food detector camera"
        >

        <p class="note">
            Hold one food item near the center of the camera.
        </p>
    </main>
</body>
</html>
"""


def generate_frames():
    frame_number = 0
    display_label = "Starting..."
    display_confidence = 0.0

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        frame_number += 1

        # Only run the neural network every third frame
        # so the video stays smoother.
        if frame_number % PREDICT_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            resized = cv2.resize(
                rgb_frame,
                IMAGE_SIZE
            )

            image_array = resized.astype(np.float32)
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            raw_score = float(
                model.predict(
                    image_array,
                    verbose=0
                )[0][0]
            )

            # Model output:
            # 0 = healthy
            # 1 = unhealthy
            if raw_score >= 0.5:
                predicted_index = 1
                confidence = raw_score
            else:
                predicted_index = 0
                confidence = 1.0 - raw_score

            if confidence >= CONFIDENCE_THRESHOLD:
                display_label = class_names[predicted_index]
            else:
                display_label = "uncertain"

            display_confidence = confidence * 100

        if display_label == "healthy":
            color = (0, 255, 0)
        elif display_label == "unhealthy":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(
            frame,
            (10, 10),
            (440, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {display_label}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {display_confidence:.1f}%",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not encoded:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Camera opened successfully.")
    print("Open the displayed address on your laptop.")

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True
        )
    finally:
        camera.release()
        print("Camera closed.")
