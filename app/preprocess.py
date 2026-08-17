from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy import stats
from scipy.fft import rfft

BASELINE = 128.0
FEATURE_COLUMNS = [
    "peak_abs_amplitude",
    "energy_centered",
    "rms_centered",
    "zero_crossing_rate",
    "spectral_centroid_bin",
    "crest_factor",
    "attack_energy_ratio",
]


def read_waveform_from_text(raw_text: str) -> np.ndarray:
    """Lit une forme d'onde brute depuis le contenu texte d'un fichier .txt."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Le fichier ne contient pas assez de données pour être interprété.")

    tokens = lines[1].split()
    try:
        signal = np.array([int(token, 16) for token in tokens], dtype=float)
    except ValueError:
        signal = np.array([float(token) for token in tokens], dtype=float)

    if signal.size == 0:
        raise ValueError("Aucune donnée détectée dans le fichier.")

    if signal.size != 512:
        # On accepte des longueurs proches de 512, car certaines acquisitions peuvent varier légèrement.
        # Pour l'application, on garde l'hypothèse de 512 points pour le modèle entraîné.
        if signal.size < 256:
            raise ValueError(f"Taille de signal invalide : {signal.size} points. Le modèle attend 512 points.")

    return signal[:512]


def read_waveform_from_file(path: str | Path) -> np.ndarray:
    """Charge un signal brut depuis un fichier texte."""
    file_path = Path(path)
    return read_waveform_from_text(file_path.read_text(encoding="utf-8", errors="replace"))


def extract_features(signal: Iterable[float]) -> Dict[str, float]:
    """Calcule les 7 features utilisées par le modèle SVM final."""
    signal_array = np.asarray(list(signal), dtype=float)
    if signal_array.size == 0:
        raise ValueError("Le signal est vide.")

    if signal_array.size < 512:
        # on complète par répétition si nécessaire pour rester compatible avec le résultat attendu.
        padding = np.full(512 - signal_array.size, signal_array[-1] if signal_array.size else BASELINE)
        signal_array = np.concatenate([signal_array, padding])

    signal_array = signal_array[:512]
    centered = signal_array - BASELINE
    n = signal_array.size

    peak_idx = int(np.argmax(np.abs(centered)))
    peak_val = centered[peak_idx]

    signs = np.sign(centered)
    signs[signs == 0] = 1
    zero_crossings = int(np.sum(signs[:-1] != signs[1:]))

    half = n // 2
    energy_first = float(np.sum(centered[:half] ** 2))
    energy_second = float(np.sum(centered[half:] ** 2))
    energy_total = energy_first + energy_second
    attack_ratio = energy_first / energy_total if energy_total > 0 else np.nan

    spectrum = np.abs(rfft(centered))
    freqs_bin = np.arange(len(spectrum))
    spec_sum = float(np.sum(spectrum))

    spectral_centroid = float((freqs_bin * spectrum).sum() / spec_sum) if spec_sum > 0 else np.nan
    dominant_bin = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0

    n3 = len(spectrum) // 3
    energy_low = float(np.sum(spectrum[:n3] ** 2))
    energy_mid = float(np.sum(spectrum[n3 : 2 * n3] ** 2))
    energy_high = float(np.sum(spectrum[2 * n3 :] ** 2))
    spec_energy_total = energy_low + energy_mid + energy_high

    rms = float(np.sqrt(np.mean(centered ** 2)))
    crest_factor = float(np.abs(peak_val) / (rms + 1e-9))

    features = {
        "peak_abs_amplitude": float(np.abs(peak_val)),
        "energy_centered": float(np.sum(centered ** 2)),
        "rms_centered": rms,
        "zero_crossing_rate": zero_crossings / n,
        "spectral_centroid_bin": spectral_centroid,
        "crest_factor": crest_factor,
        "attack_energy_ratio": float(attack_ratio),
    }

    for key in features:
        if np.isnan(features[key]):
            features[key] = 0.0

    return features


def feature_vector_from_signal(signal: Iterable[float]) -> np.ndarray:
    """Construit un vecteur NumPy ordonné selon les colonnes de features du modèle."""
    features = extract_features(signal)
    return np.array([features[col] for col in FEATURE_COLUMNS], dtype=float)


def feature_vector_from_file(path: str | Path) -> np.ndarray:
    signal = read_waveform_from_file(path)
    return feature_vector_from_signal(signal)
