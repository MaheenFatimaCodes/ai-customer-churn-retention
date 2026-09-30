# Churn Prediction API — Docker image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Model artifacts (models/*.pkl) must already exist — run src/train_models.py,
# src/explain.py, and src/segmentation.py once locally before building the image,
# or mount a volume with pre-trained models at /app/models.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
