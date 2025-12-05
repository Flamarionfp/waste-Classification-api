import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


# ===== SPLIT =====
df_train, df_test = train_test_split(
    df,
    test_size=0.01,     
    stratify=df["label"],
    random_state=42
)

X_train, y_train = load_images_to_array(df_train)
X_test, y_test = load_images_to_array(df_test)


# ===== MODELO =====
# Esse bloco cria um pipeline que primeiro normaliza os dados e 
# depois treina uma rede neural MLP robusta, com três camadas 
# grandes (1024 → 512 → 256 neurônios), usando ReLU para acelerar o 
# aprendizado. O modelo treina por bastante tempo, mas conta com regularização e 
# early stopping para evitar overfitting. A taxa de aprendizado é moderada, o batch 
# é relativamente grande, e o treinamento para automaticamente quando percebe que não está 
# mais evoluindo.
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256),
        activation="relu",
        max_iter=20000,
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        early_stopping=True,
        n_iter_no_change=20
    )
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

print("Modelo treinado e salvo em:", MODEL_PATH)
