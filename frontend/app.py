
import os
from io import BytesIO

import streamlit as st
import requests
import numpy as np
import pydicom


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩻",
    layout="wide"
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:5000"
)

CLASS_NAMES = [
    "No Lung Opacity / Not Normal",
    "Normal",
    "Lung Opacity"
]


# --------------------------------------------------
# Application title and description
# --------------------------------------------------

st.title("Chest X-ray Classification")

st.write(
    "An experimental three-class chest X-ray "
    "classification prototype using EfficientNetB0."
)


# --------------------------------------------------
# Backend connection status
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{BACKEND_URL}/health",
        timeout=10
    )

    if health_response.status_code == 200:
        st.success("Prediction backend is connected.")
    else:
        st.warning("Prediction backend is not responding correctly.")

except requests.RequestException:
    st.error(
        "Cannot connect to the prediction backend. "
        "Please check that the backend service is running."
    )


# --------------------------------------------------
# DICOM image preview
# --------------------------------------------------

def display_dicom_preview(file_bytes):

    dicom_data = pydicom.dcmread(
        BytesIO(file_bytes)
    )

    image = dicom_data.pixel_array.astype(np.float32)

    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        image = image.max() - image

    # Scale only for visual display.
    # Model preprocessing is performed by the backend.
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image = (
            (image - image_min) /
            (image_max - image_min) * 255
        )

    return image.astype(np.uint8)


# --------------------------------------------------
# Upload interface
# --------------------------------------------------

st.subheader("Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Upload a DICOM file",
    type=["dcm"]
)


# --------------------------------------------------
# Image preview and prediction
# --------------------------------------------------

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    try:
        display_image = display_dicom_preview(file_bytes)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                display_image,
                caption="Uploaded Chest X-ray",
                use_container_width=True
            )

        with col2:
            st.subheader("Model Prediction")

            if st.button(
                "Run Prediction",
                type="primary"
            ):

                try:
                    with st.spinner("Processing X-ray..."):

                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    file_bytes,
                                    "application/dicom"
                                )
                            },
                            timeout=180
                        )

                    if response.status_code == 200:

                        result = response.json()

                        st.success(
                            f"Predicted Class: "
                            f"{result['predicted_label']}"
                        )

                        st.write("Class probabilities:")

                        probabilities = result["probabilities"]

                        for class_name in CLASS_NAMES:

                            probability = probabilities[class_name]

                            st.write(
                                f"{class_name}: "
                                f"{probability * 100:.2f}%"
                            )

                            st.progress(
                                float(probability)
                            )

                    else:
                        error_message = response.json().get(
                            "error",
                            "Prediction request failed."
                        )

                        st.error(error_message)

                except requests.RequestException as error:
                    st.error(
                        f"Unable to reach prediction backend: {error}"
                    )

    except Exception as error:
        st.error(
            f"Unable to display this DICOM file: {error}"
        )


# --------------------------------------------------
# Educational disclaimer
# --------------------------------------------------

st.divider()

st.warning(
    "**Educational prototype only.** This model is not "
    "clinically validated and must not be used for diagnosis, "
    "treatment decisions, or patient management. Predictions "
    "may be incorrect, including missed Lung Opacity cases."
)

st.caption(
    "Model: EfficientNetB0 | Input: 224 × 224 × 3 | "
    "Classes: 3 | Experimental prototype"
)
