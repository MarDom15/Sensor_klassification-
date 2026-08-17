from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from app.preprocess import FEATURE_COLUMNS, feature_vector_from_signal, read_waveform_from_file

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "shot_vs_parasite_svm.joblib"

LABELS = {1: "Treffer", 0: "Parasite"}


def load_model_artifact() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_signal_array(signal: np.ndarray) -> Dict[str, Any]:
    """Prédit la classe d'un signal brut donné en tableau NumPy."""
    artifact = load_model_artifact()
    model = artifact["pipeline"]

    feature_vector = feature_vector_from_signal(signal)
    feature_vector = feature_vector.reshape(1, -1)

    prediction = int(model.predict(feature_vector)[0])
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = float(np.max(probabilities))

    return {
        "prediction": prediction,
        "label": LABELS.get(prediction, "Inconnu"),
        "confidence": confidence,
        "probabilities": {
            "Parasite": float(probabilities[0]),
            "Treffer": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
        },
        "feature_columns": FEATURE_COLUMNS,
    }


def predict_signal_file(path: str | Path) -> Dict[str, Any]:
    signal = read_waveform_from_file(path)
    return predict_signal_array(signal)
