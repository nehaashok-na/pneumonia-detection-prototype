FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system libraries required by TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r /app/backend/requirements.txt

# Copy backend application and trained model
COPY backend/app.py /app/backend/app.py
COPY final_efficientnetb0_model.keras /app/final_efficientnetb0_model.keras
COPY model_config.json /app/model_config.json

EXPOSE 5000

# Start Flask through Gunicorn
CMD ["gunicorn", "--chdir", "/app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "180", "backend.app:app"]
