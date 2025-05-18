import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from PIL import Image
from tqdm import tqdm

# Directories
DATASET_DIR = "dataset"
MODEL_DIR = "models"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# CNN MODEL TRAINING
# -----------------------------
print("✅ Training CNN model...")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

cnn_model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

cnn_model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
print("✅ CNN model saved!")

# -----------------------------
# RANDOM FOREST MODEL TRAINING
# -----------------------------
print("✅ Training Random Forest model...")

X = []
y = []
label_map = {label: idx for idx, label in enumerate(os.listdir(DATASET_DIR))}

for label in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for img_file in tqdm(os.listdir(class_dir), desc=f"Loading {label}"):
        try:
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
            img_arr = np.array(img).flatten()
            X.append(img_arr)
            y.append(label_map[label])
        except:
            continue

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
print("✅ RF model saved!")
