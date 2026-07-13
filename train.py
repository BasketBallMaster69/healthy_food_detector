from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# -----------------------------
# Project settings
# -----------------------------
DATASET_PATH = Path("dataset")
MODEL_PATH = Path("models") / "food_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load the dataset
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

class_names = train_dataset.class_names

print("\nClass names:", class_names)
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))

# A small fixed prefetch is safer for Jetson memory
train_dataset = train_dataset.prefetch(1)
validation_dataset = validation_dataset.prefetch(1)

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# -----------------------------
# Load MobileNetV2
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze the pretrained layers
base_model.trainable = False

# -----------------------------
# Build the classifier
# -----------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 expects pixel values scaled from -1 to 1
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# One output neuron because this is binary classification
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# -----------------------------
# Compile the model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Training callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    ),
]

# -----------------------------
# Train the model
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# -----------------------------
# Final evaluation
# -----------------------------
validation_loss, validation_accuracy = model.evaluate(
    validation_dataset,
    verbose=1,
)

print(f"\nValidation loss: {validation_loss:.4f}")
print(f"Validation accuracy: {validation_accuracy:.4f}")

# Save class labels
Path("labels.txt").write_text("\n".join(class_names) + "\n")

# Save the final model
model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Labels saved to: labels.txt")