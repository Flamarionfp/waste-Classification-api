FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training.py /app/training.py
COPY api.py /app/api.py
COPY class_mapping.py /app/class_mapping.py

COPY dataset/ /app/dataset/

ENV DATASET_DIR=/app/dataset
ENV MODEL_PATH=/app/waste_classifier.pkl

RUN python training.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
