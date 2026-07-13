from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/food_model.keras")
LABELS_PATH = Path("labels.txt")
IMAGE_SIZE = (224, 224)

if len(sys.argv) != 2:
    print("Usage: python3 predict.py path/to/image.jpg")
    raise SystemExit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"Image not found: {image_path}")
    raise SystemExit(1)

if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    line.strip()
    for line in LABELS_PATH.read_text().splitlines()
    if line.strip()
]

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE,
)

image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

prediction = float(model.predict(image_array, verbose=0)[0][0])

# The model learned:
# 0 = healthy
# 1 = unhealthy
if prediction >= 0.5:
    label = class_names[1]
    confidence = prediction
else:
    label = class_names[0]
    confidence = 1.0 - prediction

print(f"\nPrediction: {label}")
print(f"Confidence: {confidence * 100:.2f}%")
print(f"Raw model score: {prediction:.4f}")