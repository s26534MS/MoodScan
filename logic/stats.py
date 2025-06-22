from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.metrics import confusion_matrix

COLS = [
    "czas", "plik", "model",
    "emocja", "trafiony",
    "etykieta_uzytkownika", "score"
]


def model_accuracy(df: pd.DataFrame) -> pd.Series:
    """
    Accuracy per-model (trafiony==1) dla przekazanego df.
    Wynik posortowany malejąco (% w skali 0-1).
    """
    if df.empty:
        return pd.Series(dtype=float)

    return (
        df.groupby("model")["trafiony"]
        .mean()
        .sort_values(ascending=False)
    )


def confusion(
        df: pd.DataFrame,
        model: Optional[str] = None
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Zwraca (macierz, etykiety).
    • model=None  → łączna confusion-matrix.
    • model="DeepFace"…  → tylko wskazany model.

    Gdy brak pełnych etykiet: (None, []).
    """
    if model is not None:
        df = df[df["model"] == model]

    if "etykieta_uzytkownika" not in df.columns:
        return None, []

    mask = df["etykieta_uzytkownika"].notna() & df["emocja"].notna()
    if not mask.any():
        return None, []

    y_true = df.loc[mask, "etykieta_uzytkownika"].str.capitalize()
    y_pred = df.loc[mask, "emocja"].str.capitalize()

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def read_csv(path: str) -> pd.DataFrame:
    """
    Ładuje wskazany plik historii i gwarantuje układ kolumn = COLS.
    """
    df = pd.read_csv(path, engine="python")
    for col in COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLS]
