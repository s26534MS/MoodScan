import argparse
import cv2
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Flatten, Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
C2I = {c: i for i, c in enumerate(CLASSES)}

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(path, size=48):
    """Zwraca wykadrowaną i przeskalowaną twarz lub None, gdy brak twarzy."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y: y + h, x: x + w]
    face = cv2.resize(face, (size, size))
    return face


def load_dataset(base, img_size, limit=None):
    """Ładuje folder z podfolderami klas.  Zwraca listę (img, label)."""
    data = []
    for cls in CLASSES:
        folder = os.path.join(base, cls)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if limit and len(data) >= limit:
                break
            face = crop_face(os.path.join(folder, fname), img_size)
            if face is not None:
                data.append((face, C2I[cls]))
    return data


ap = argparse.ArgumentParser()
ap.add_argument("--img-size", type=int, default=48,
                help="output size (48 lub 96)")
ap.add_argument("--kdef", default="../kdef_sorted",
                help="główny katalog KDEF")
ap.add_argument("--fer", default="../dane", help="root FER-2013 train/test")
ap.add_argument("--with-fer", action="store_true",
                help="dorzuca FER-2013 (filtrowane jakościowo)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--gpu", action="store_true", help="nie wymusza CPU")
args = ap.parse_args()

print("Ładuję KDEF …")
kdef = load_dataset(args.kdef, args.img_size)
print(f"✓  znaleziono {len(kdef)} przykl. twarzy z KDEF")

data = kdef

if args.with_fer:
    print("Ładuję FER-2013 …")
    fer_train = load_dataset(os.path.join(args.fer, "train"), args.img_size, limit=6000)
    fer_test = load_dataset(os.path.join(args.fer, "test"), args.img_size, limit=1000)
    print(f"✓  dodano {len(fer_train) + len(fer_test)} przykl. z FER")
    data += fer_train + fer_test

cnt = Counter([lbl for _, lbl in data])
print("Rozkład klas:", {CLASSES[i]: n for i, n in cnt.items()})

np.random.shuffle(data)
X = np.array([d[0] for d in data]).reshape(-1, args.img_size, args.img_size, 1) / 255.0
y = to_categorical([d[1] for d in data], num_classes=len(CLASSES))

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42,
                                        stratify=y)

weights = compute_class_weight("balanced",
                               classes=np.arange(len(CLASSES)),
                               y=[d[1] for d in data])
class_weights = {i: w for i, w in enumerate(weights)}
print("Class-weights:", class_weights)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)
datagen.fit(Xtr)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(args.img_size, args.img_size, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=4, factor=0.3, verbose=1),
    ModelCheckpoint("model_kdef_v2.h5", monitor="val_accuracy",
                    save_best_only=True, verbose=1)
]

print("🚀 Trening …")
hist = model.fit(
    datagen.flow(Xtr, ytr, batch_size=args.batch),
    epochs=args.epochs,
    validation_data=(Xval, yval),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"], label="train")
plt.plot(hist.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("history.png", dpi=120)
plt.show()
