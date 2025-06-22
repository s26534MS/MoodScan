from __future__ import annotations
import cv2
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from functools import lru_cache
from pathlib import Path

from logic import image_utils

MODEL_PATHS = {
    "FER-2013": "models/best_model.h5",
    "KDEF": "models/model_kdef_v2.h5",
}

OWN_CLASSES = ["angry", "disgust", "fear",
               "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48


@lru_cache(maxsize=1)
def _deep_analyze(path: str | Path):
    return DeepFace.analyze(img_path=str(path),
                            actions=["emotion"],
                            enforce_detection=False)[0]


def predict_deepface(path: str | Path):
    res = _deep_analyze(path)
    emo = res["dominant_emotion"]
    dist = res["emotion"]
    return emo, dist


@lru_cache(maxsize=None)
def _load(path: str | Path):
    return load_model(str(path))


def predict_cnn(path: str | Path, model_label: str, crop: bool = True):
    """
    Zwraca (dominująca_emo, lista_procentów[7])
    """
    if model_label not in MODEL_PATHS:
        raise KeyError(model_label)

    net = _load(MODEL_PATHS[model_label])

    if crop:
        face = image_utils.crop_face(path, size=IMG_SIZE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        face = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA).astype("float32") / 255.0

    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = net.predict(face, verbose=0)[0]
    idx = int(np.argmax(probs))
    emo = OWN_CLASSES[idx]
    return emo, probs
