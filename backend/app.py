
import os
import json
import logging
from io import BytesIO

import numpy as np
import tensorflow as tf
import pydicom

from flask import Flask, request, jsonify
from skimage.transform import resize
from werkzeug.exceptions import RequestEntityTooLarge


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "final_efficientnetb0_model.keras"
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR,
    "model_config.json"
)


# --------------------------------------------------
# 2. Load configuration and trained model
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

CLASS_NAMES = config["class_names"]
IMG_SIZE = tuple(config["input_size"])

# Provisional threshold from the current configuration.
# Recalibration against the recovered model's validation
# predictions is still pending.
LUNG_OPACITY_THRESHOLD = float(
    config["lung_opacity_threshold"]
)

if len(CLASS_NAMES) != 3:
    raise ValueError("Expected exactly three class names.")

if len(IMG_SIZE) != 2:
    raise ValueError("Expected a two-dimensional input size.")

model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError("Model output does not match the class configuration.")

app.logger.info("EfficientNetB0 model loaded successfully.")


# --------------------------------------------------
# 3. DICOM preprocessing
# --------------------------------------------------

def preprocess_dicom(file_bytes):

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    # Read the uploaded DICOM file.
    dicom_data = pydicom.dcmread(BytesIO(file_bytes))

    if not hasattr(dicom_data, "PixelData"):
        raise ValueError("The DICOM file does not contain image pixels.")

    # Extract image pixel array.
    image = dicom_data.pixel_array.astype(np.float32)

    # This application supports single-frame 2D images only.
    if image.ndim != 2:
        raise ValueError(
            "Unsupported image dimensions. Please upload a single-frame 2D DICOM."
        )

    if not np.isfinite(image).all():
        raise ValueError("The image contains invalid pixel values.")

    # Correct MONOCHROME1 images.
    if getattr(dicom_data, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    # Preserve the preprocessing pipeline used for the model.
    image = np.clip(image, 0, 255)

    image = resize(
        image,
        IMG_SIZE,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    # Replicate grayscale into three channels.
    image = np.stack([image, image, image], axis=-1)

    # Keep pixel values in the 0-255 range.
    image = image.astype(np.float32)

    # Add batch dimension.
    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

def predict_xray(image_batch):

    probabilities = model.predict(
        image_batch,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError("Unexpected number of model output probabilities.")

    if not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid prediction values.")

    # Choose the highest probability among classes 0 and 1.
    predicted_class = int(np.argmax(probabilities[:2]))

    # Apply the currently provisional Lung Opacity threshold.
    if probabilities[2] >= LUNG_OPACITY_THRESHOLD:
        predicted_class = 2

    return predicted_class, probabilities


# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": config["model_name"]
    }), 200


# --------------------------------------------------
# 6. Prediction API endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No DICOM file was uploaded."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "No file was selected."
        }), 400

    try:
        file_bytes = uploaded_file.read()

        # Validate and preprocess the uploaded DICOM.
        image_batch = preprocess_dicom(file_bytes)

    except Exception:
        app.logger.exception("DICOM validation or preprocessing failed.")

        return jsonify({
            "error": "Unable to read this file as a supported DICOM image."
        }), 400

    try:
        # Generate prediction.
        predicted_class, probabilities = predict_xray(image_batch)

        response = {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_NAMES[predicted_class],
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 6)
                for i in range(len(CLASS_NAMES))
            }
        }

        return jsonify(response), 200

    except Exception:
        app.logger.exception("Model prediction failed.")

        return jsonify({
            "error": "Prediction could not be completed. Please try again."
        }), 500


# --------------------------------------------------
# 7. Upload size error handler
# --------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):

    return jsonify({
        "error": "The uploaded file exceeds the 20 MB limit."
    }), 413


# --------------------------------------------------
# 8. Local development entry point
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
