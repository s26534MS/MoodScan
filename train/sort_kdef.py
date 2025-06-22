import os
import shutil

SOURCE_DIR = "KDEF"
DEST_DIR = "../kdef_sorted"

EMOTION_MAP = {
    "AF": "fear",
    "AN": "angry",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise"
}

os.makedirs(DEST_DIR, exist_ok=True)
for emotion in EMOTION_MAP.values():
    os.makedirs(os.path.join(DEST_DIR, emotion), exist_ok=True)

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.lower().endswith(".jpg") and len(file) >= 7:
            emotion_code = file[4:6]
            view = file[6]

            if view != "S":
                continue

            if emotion_code in EMOTION_MAP:
                emotion = EMOTION_MAP[emotion_code]
                source_path = os.path.join(root, file)
                dest_path = os.path.join(DEST_DIR, emotion, file)

                shutil.copy2(source_path, dest_path)

print("✅ Zdjęcia zostały posortowane do folderu:", DEST_DIR)
