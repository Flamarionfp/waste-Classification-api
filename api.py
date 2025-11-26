import io
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from class_mapping import CLASS_TRANSLATIONS

app = FastAPI(title="Waste Classification API")

# Carregar o modelo
saved = joblib.load("waste_classifier.pkl")
model = saved["model"]
label_encoder = saved["label_encoder"]
IMG_SIZE = saved["img_size"]

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.flatten().reshape(1, -1)

    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        bytes_data = await file.read()
        X = preprocess_image(bytes_data)

        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0].max())

        class_name = label_encoder.inverse_transform([pred])[0]
        class_name_pt = CLASS_TRANSLATIONS.get(class_name, class_name)

        return JSONResponse({
            "predicted_class": class_name,       
            "predicted_class_pt": class_name_pt,
            "confidence": proba
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/classes")
def list_classes():
    items = []

    for key, label in CLASS_TRANSLATIONS.items():
        items.append({
            "id": key,       
            "label": label
        })

    return JSONResponse(content={"classes": items})

@app.get("/")
def health_check():
    return JSONResponse(content={"status": "UP"})