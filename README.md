# Healthy Food Detector using NVIDIA Jetson Orin Nano

 VIDEO HERE https://drive.google.com/file/d/16eXiu5r0BD1qAzsIllrN0xWUM_SHFhXH/view?usp=sharing

This project uses machine learning and computer vision to classify food
as **healthy** or **unhealthy** in real time. A webcam connected to an
NVIDIA Jetson Orin Nano captures live video, and a Convolutional Neural
Network (CNN) analyzes each frame. The prediction and confidence score
are displayed in a web browser, allowing users to instantly determine
whether the detected food is healthy or unhealthy.

The project was built using Python, TensorFlow, OpenCV, Flask, and the
NVIDIA Jetson Orin Nano with GPU acceleration. The model was trained
using images from the Food-101 dataset and achieved a validation
accuracy of approximately **93%**.

**Image:** *(Insert a screenshot of your running web application here.)*

# The Algorithm

This project uses a **Convolutional Neural Network (CNN)**, which is a
deep learning algorithm designed specifically for recognizing images.

The project works in several steps:

## Step 1: Collect the Data

The model was trained using the Food-101 dataset. Instead of training on
all 101 food categories, twelve categories were selected and grouped
into two classes.

### Healthy Foods

-   Caesar Salad
-   Greek Salad
-   Grilled Salmon
-   Edamame
-   Seaweed Salad
-   Sushi

### Unhealthy Foods

-   Pizza
-   Hamburger
-   French Fries
-   Donuts
-   Ice Cream
-   Chicken Wings

Each category contains approximately 1,000 images, giving the model
thousands of examples to learn from.

## Step 2: Build the Dataset

A Python script automatically copied the selected Food-101 images into
two folders:

-   `dataset/healthy`
-   `dataset/unhealthy`

This created a clean dataset containing only the foods needed for this
project.

## Step 3: Train the Model

TensorFlow automatically loaded all images from the dataset.

During training:

-   Images were resized to **224 × 224 pixels**
-   The images were divided into training and validation sets
-   A CNN learned patterns such as colors, textures, and shapes
-   The Jetson Orin Nano GPU accelerated the training process using CUDA

The model was trained for ten epochs and automatically saved the
best-performing version.

### Training Results

-   Training Images: **9,600**
-   Validation Images: **2,400**
-   Classes: **Healthy** and **Unhealthy**
-   Validation Accuracy: **92.96%**

## Step 4: Real-Time Prediction

After training, the saved model is loaded into memory.

When the webcam captures a frame:

1.  The image is resized to 224 × 224 pixels.
2.  Pixel values are normalized.
3.  The CNN predicts whether the food is healthy or unhealthy.
4.  The prediction confidence is calculated.
5.  The result is displayed on a live webpage.

The webpage continuously updates as new frames are received from the
camera.

# Running this Project

## Requirements

### Hardware

-   NVIDIA Jetson Orin Nano
-   USB Webcam
-   JetPack 6.0

### Software

-   Python 3.10
-   TensorFlow 2.16
-   OpenCV
-   Flask
-   NumPy
-   Pillow

Install the required libraries:

``` bash
pip3 install tensorflow==2.16.1 flask opencv-python numpy pillow
```

## Training the Model

``` bash
cd ~/healthy_food_detector
python3 train.py
```

The trained model will automatically be saved as:

`models/food_model.keras`

## Testing a Single Image

``` bash
python3 predict.py path/to/image.jpg
```

Example:

``` bash
python3 predict.py dataset/healthy/caesar_salad/2668462.jpg
```

## Running the Live Camera

``` bash
cd ~/healthy_food_detector
python3 web_camera.py
```

Find the Jetson's IP:

``` bash
hostname -I
```

Open a browser on the same network:

`http://YOUR_JETSON_IP:5000`

# Project Files

-   **build_dataset.py** -- Copies Food-101 images into Healthy and
    Unhealthy folders.
-   **train.py** -- Trains the CNN.
-   **predict.py** -- Predicts a single image.
-   **web_camera.py** -- Runs the live webcam application.
-   **models/food_model.keras** -- Saved trained model.
-   **labels.txt** -- Stores the class names.

# Technologies Used

-   Python
-   TensorFlow / Keras
-   OpenCV
-   Flask
-   NumPy
-   NVIDIA CUDA
-   NVIDIA Jetson Orin Nano
-   Food-101 Dataset

# Results

The final model achieved approximately **93% validation accuracy** while
classifying foods into two categories: healthy and unhealthy.

The application performs real-time predictions using the Jetson Orin
Nano GPU and displays the results through a web interface.

# Future Improvements

-   Recognize the exact food item instead of only healthy or unhealthy.
-   Detect multiple foods in the same image.
-   Draw colored boxes around detected food.
-   Add voice feedback.
-   Increase the number of food categories.
-   Optimize the model with TensorRT.
-   Improve accuracy with additional datasets.

# Video Demonstration

*(Insert your YouTube or Google Drive video link here.)*
