# Chest X-ray Classification – Pneumonia Detection Prototype

## 1. Project Overview

This project is an experimental three-class chest X-ray classification application using a fine-tuned EfficientNetB0 model.

The application predicts one of three classes:

1. No Lung Opacity / Not Normal
2. Normal
3. Lung Opacity

**Disclaimer:** This is an educational prototype. It is not clinically validated and must not be used for diagnosis, treatment decisions, or patient management.

## 2. Application Architecture

- Frontend: Streamlit application for DICOM upload and displaying predictions.
- Backend: Flask API for model loading, DICOM preprocessing, and inference.
- Model: Fine-tuned EfficientNetB0.
- Containerization: Docker and Docker Compose.
- Deployment environment: GitHub Codespaces.

The frontend communicates with the backend through HTTP requests. Docker Compose manages both services. The frontend waits for the backend health check before starting.

## 3. Project Structure

    pneumonia-detection/
    ├── backend/
    │   ├── app.py
    │   ├── requirements.txt
    │   └── Dockerfile
    ├── frontend/
    │   ├── app.py
    │   ├── requirements.txt
    │   └── Dockerfile
    ├── .devcontainer/
    │   └── devcontainer.json
    ├── .gitignore
    ├── .python-version
    ├── final_efficientnetb0_model.keras
    ├── model_config.json
    ├── docker-compose.yml
    └── README.md

## 4. Model Input and Preprocessing

- Input image size: 224 × 224 pixels.
- Input channels: 3.
- Input format: DICOM.
- Supports single-frame, two-dimensional DICOM images.
- MONOCHROME1 images are inverted when required.
- Pixel values are clipped to 0–255.
- Images are resized while preserving intensity range.
- Grayscale images are replicated into three channels.
- Pixel values remain in the 0–255 range.

The backend accepts uploaded files up to 20 MB.

## 5. Running with Docker

Ensure Docker and Docker Compose are available.

Build and start both services:

    docker compose up --build

Open the Streamlit application at:

    http://localhost:8501

Service ports:

| Component | Port |
|---|---:|
| Streamlit frontend | 8501 |
| Flask backend | 5000 |

The backend health endpoint is used to check service readiness.

Stop the services:

    docker compose down

## 6. Backend API

### Health check

    GET /health

Returns the backend health status and model name.

### Prediction

    POST /predict

Send a DICOM file using multipart form data with the field name `file`.

Example response:

    {
      "predicted_class": 2,
      "predicted_label": "Lung Opacity",
      "probabilities": {
        "No Lung Opacity / Not Normal": 0.12,
        "Normal": 0.18,
        "Lung Opacity": 0.70
      }
    }

The values above are illustrative and do not represent an actual model prediction.

## 7. GitHub Codespaces Deployment

1. Push the project files to a GitHub repository.
2. Open the repository on GitHub.
3. Select Code → Codespaces → Create codespace.
4. Allow Codespaces to build the Docker environment.
5. Wait for the backend health check and frontend service to become ready.
6. Open the forwarded port 8501.
7. Upload a supported DICOM file and verify the prediction interface.

The application must undergo an actual Docker build and inference test before deployment can be considered successful.

## 8. Model and Validation Limitations

- The model is experimental.
- Development used a limited dataset subset.
- External dataset performance has not been established.
- Predictions may be incorrect, including missed Lung Opacity cases.
- The current Lung Opacity threshold is provisional.
- The threshold has not yet been recalibrated against validation predictions from the recovered model.
- Deployment functionality does not establish clinical validity.

## 9. Responsible Use

This application is intended solely for educational demonstration and machine-learning experimentation.

It is not a medical device and must not be used to make clinical decisions.
