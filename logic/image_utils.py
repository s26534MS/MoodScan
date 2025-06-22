from __future__ import annotations

import cv2
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np


def load_photo(path: str | Path, max_side: int = 560) -> ImageTk.PhotoImage:
    """Ładuje plik, skaluje tak, aby dłuższy bok = max_side px, zwraca PhotoImage."""
    img = Image.open(path)
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)),
                         Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


def crop_face(path: str | Path,
              size: int = 48) -> np.ndarray:
    """
    Zwraca wycinek (ROI) twarzy: tablica (size×size),
    gotową do /255 i reshape(1, size, size, 1).

    • Szukamy **największej** twarzy (Haar Cascade).
    • Gdy brak detekcji → fallback: centralny kwadrat kadru.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60))

    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
    else:
        h0, w0 = gray.shape
        side = min(h0, w0)
        y0 = (h0 - side) // 2
        x0 = (w0 - side) // 2
        face = gray[y0:y0 + side, x0:x0 + side]

    face = cv2.resize(face, (size, size),
                      interpolation=cv2.INTER_AREA)

    face = face.astype("float32") / 255.0
    return face
