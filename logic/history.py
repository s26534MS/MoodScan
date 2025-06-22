import os, datetime
import pandas as pd
from typing import Optional

SCORE = {"happy": 1, "surprise": 0, "neutral": 0,
         "sad": -1, "angry": -1, "disgust": -1, "fear": -1}

COLS = ["czas", "plik", "model", "emocja",
        "trafiony", "etykieta_uzytkownika", "score"]


def _csv(user: str) -> str:
    return f"historia_{user}.csv"


def _write(df: pd.DataFrame, user: str) -> None:
    file = _csv(user)
    need_header = (not os.path.exists(file)) or os.path.getsize(file) == 0
    df.to_csv(file, mode="a", header=need_header, index=False)


def _read(user: str) -> pd.DataFrame:
    file = _csv(user)
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(file, names=COLS, header=0, engine="python")
    return df[COLS]


def save_row(path: str, model: str, emo_pred: str,
             user: str, correct: bool = False) -> None:
    _write(pd.DataFrame([{
        "czas": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plik": os.path.basename(path),
        "model": model,
        "emocja": emo_pred,
        "trafiony": int(correct),
        "etykieta_uzytkownika": pd.NA,
        "score": SCORE.get(emo_pred.lower(), 0)
    }]), user)


def update_label(path: str, model: str, user_label: str,
                 user: str, correct: bool) -> bool:
    df = _read(user)
    mask = (df["plik"] == os.path.basename(path)) & (df["model"] == model)
    if not mask.any():
        return False
    idx = mask.idxmax()
    df.loc[idx, "etykieta_uzytkownika"] = user_label.capitalize()
    df.loc[idx, "trafiony"]             = int(correct)
    df.to_csv(_csv(user), index=False)
    return True


def load_stats(user: str) -> Optional[pd.DataFrame]:
    df = _read(user)
    return None if df.empty else df
