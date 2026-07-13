import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# Settings
# -----------------------------
CAMERA_INDEX = 0
IMAGE_SIZE = 224
MODEL_PATH = "models/food_model.keras"
LABELS_PATH = "labels.txt"

# -----------------------------
# Load model and labels
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file.readlines()
        if line.strip()
    ]

print("Classes:", class_names)

# -----------------------------
# Open webcam
# -----------------------------
camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open /dev/video0")

print("Camera opened. Press Q to quit.")

# -----------------------------
# Live camera loop
# -----------------------------
while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # OpenCV reads images as BGR.
    # TensorFlow trained using RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = cv2.resize(
        rgb_frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = resized_frame.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    raw_score = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # 0 = healthy
    # 1 = unhealthy
    if raw_score >= 0.5:
        label = class_names[1]
        confidence = raw_score
        color = (0, 0, 255)
    else:
        label = class_names[0]
        confidence = 1.0 - raw_score
        color = (0, 255, 0)

    text = f"{label}: {confidence * 100:.1f}%"

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Healthy Food Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
camera.release()
cv2.destroyAllWindows()
print("Camera closed.")
