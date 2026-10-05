from pathlib import Path

import torch
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = BASE_DIR / "fake_news_model"

# Paper convention: 0 = Real, 1 = Fake
LABEL_MAP = {
    "Real": "Real", "Fake": "Fake",
    "REAL": "Real", "FAKE": "Fake",
    "LABEL_0": "Real", "LABEL_1": "Fake",
    "0": "Real", "1": "Fake",
}

_classifier = None


def get_classifier():
    global _classifier

    if _classifier is None:
        if not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Trained model not found at {LOCAL_MODEL_DIR}. "
                "Copy the fake_news_model folder from Colab into the "
                "project folder (next to model_loader.py)."
            )

        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            "text-classification",
            model=str(LOCAL_MODEL_DIR),
            tokenizer=str(LOCAL_MODEL_DIR),
            device=device,
            truncation=True,
            max_length=256,
        )

    return _classifier


def predict_text(text):
    if not text or not text.strip():
        return {"label": "Unknown", "score": 0.0, "error": "Input text is empty"}

    try:
        classifier = get_classifier()
    except FileNotFoundError as e:
        return {"label": "Unknown", "score": 0.0, "error": str(e)}

    result = classifier(text.strip(), truncation=True, max_length=256)[0]
    raw_label = str(result["label"])

    return {
        "label": LABEL_MAP.get(raw_label, raw_label),
        "score": float(result["score"]),
        "raw_label": raw_label,
    }