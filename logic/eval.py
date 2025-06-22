import glob
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from logic import predictors

DATA_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "CK+"
MODEL_KEY = "deepface"  # tu zmiana na "kdef" lub "deepface"


def iter_ckplus(root: Path):
    """Yielduje (img_path, true_label) dla wszystkich plików w sub-folderach."""
    for emo_dir in root.iterdir():
        if emo_dir.is_dir():
            label = emo_dir.name.lower()
            for img_path in glob.glob(str(emo_dir / "*")):
                yield Path(img_path), label


def main():
    y_true, y_pred = [], []
    for img_path, label in iter_ckplus(DATA_ROOT):

        if MODEL_KEY.lower() == "deepface":
            emo, _ = predictors.predict_deepface(img_path)
        else:
            emo, _ = predictors.predict_cnn(img_path, MODEL_KEY)

        y_true.append(label)
        y_pred.append(emo.lower())

    print(classification_report(y_true, y_pred, digits=3))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
