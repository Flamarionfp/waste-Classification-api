import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
MODEL_PATH = os.getenv("MODEL_PATH", "waste_classifier.pkl")

IMG_SIZE = (32, 32)

filepaths = []
labels = []

for root, dirs, files in os.walk(DATASET_DIR):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            label = os.path.relpath(root, DATASET_DIR).split(os.sep)[0]
            filepaths.append(os.path.join(root, fname))
            labels.append(label)

df = pd.DataFrame({"filepath": filepaths, "label": labels})

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

def load_images_to_array(df):
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        img = Image.open(row["filepath"]).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        X_list.append(arr.flatten())
        y_list.append(row["label_encoded"])

    return np.array(X_list), np.array(y_list)

X, y = load_images_to_array(df)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300)
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Acurácia:", acc)

# Salvar o modelo
joblib.dump({
    "model": model,
    "label_encoder": label_encoder,
    "img_size": IMG_SIZE
}, MODEL_PATH)

print("Modelo salvo em:", MODEL_PATH)
