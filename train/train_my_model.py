import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "../dane"
IMG_SIZE = 48
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}


def load_images_from_folder(folder, label):
    data = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append((img, label))
    return data


data = []
for cls in CLASSES:
    print(f"Ładowanie klasy: {cls}")
    train_path = os.path.join(DATA_DIR, "train", cls)
    test_path = os.path.join(DATA_DIR, "test", cls)
    data += load_images_from_folder(train_path, CLASS_TO_IDX[cls])
    data += load_images_from_folder(test_path, CLASS_TO_IDX[cls])

np.random.shuffle(data)

X = np.array([x[0] for x in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
y = to_categorical(np.array([x[1] for x in data]), num_classes=len(CLASSES))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Dane przygotowane!")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

print("✅ Najlepszy model zapisany jako best_model.h5")
