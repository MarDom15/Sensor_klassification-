#!/usr/bin/env python3
"""
Leichte Inferenz fuer den Raspberry Pi: Klassifikation echter Treffer vs.
Stoersignal (Vorbei / Steinschlag / Auffahren) aus einem rohen Signal mit
512 Messpunkten (Werte 0-255, Stoss-/Vibrationssensor).

Minimale Abhaengigkeiten: numpy, scikit-learn (zum Laden der Pipeline), joblib.
Installation auf Raspberry Pi OS (vorkompilierte Pakete ueber piwheels, dort
standardmaessig eingerichtet):

    pip install numpy scikit-learn joblib

Verwendung:
    python3 rpi_inference.py pfad/zur/signaldatei.txt
    python3 rpi_inference.py --live   # Leseschleife von einem Sensor (muss angepasst werden)
"""
import os
import sys
import numpy as np
import joblib

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "models", "shot_vs_parasite_svm.joblib")
BASELINE = 128.0  # ADC-Offset (0x80) -> anpassen, falls sich Sensor/ADC aendert

FEATURES_ORDER = ["peak_abs_amplitude", "energy_centered", "rms_centered",
                   "zero_crossing_rate", "spectral_centroid_bin", "crest_factor",
                   "attack_energy_ratio"]


def extract_features(raw_values):
    """raw_values: 512 Ganzzahlen 0-255 (gleiches Format wie die Dateien ssValues*.txt)."""
    x = np.asarray(raw_values, dtype=float)
    N = len(x)
    c = x - BASELINE

    peak_idx = int(np.argmax(np.abs(c)))
    peak_val = c[peak_idx]
    peak_abs_amplitude = float(np.abs(peak_val))

    energy_centered = float(np.sum(c ** 2))
    rms_centered = float(np.sqrt(np.mean(c ** 2)))

    signs = np.sign(c)
    signs[signs == 0] = 1
    zero_crossing_rate = float(np.sum(signs[:-1] != signs[1:])) / N

    spec = np.abs(np.fft.rfft(c))
    spec_sum = spec.sum()
    freqs_bin = np.arange(len(spec))
    spectral_centroid_bin = float((freqs_bin * spec).sum() / spec_sum) if spec_sum > 0 else 0.0

    crest_factor = float(peak_abs_amplitude / (rms_centered + 1e-9))

    half = N // 2
    energy_first = float(np.sum(c[:half] ** 2))
    energy_total = energy_first + float(np.sum(c[half:] ** 2))
    attack_energy_ratio = energy_first / energy_total if energy_total > 0 else 0.0

    feats = dict(
        peak_abs_amplitude=peak_abs_amplitude,
        energy_centered=energy_centered,
        rms_centered=rms_centered,
        zero_crossing_rate=zero_crossing_rate,
        spectral_centroid_bin=spectral_centroid_bin,
        crest_factor=crest_factor,
        attack_energy_ratio=attack_energy_ratio,
    )
    return np.array([feats[f] for f in FEATURES_ORDER]).reshape(1, -1)


def load_waveform_txt(path):
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    data_line = lines[1] if len(lines) > 1 else lines[0]
    toks = data_line.split()
    return [int(t, 16) for t in toks]


def predict(raw_values, model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    X = extract_features(raw_values)
    pred = pipe.predict(X)[0]
    # Hinweis: predict_proba() des SVC ist per Platt-Scaling auf nur 74
    # Trainingsbeispielen kalibriert -> sehr instabil (in unseren Tests fast
    # konstant). Wir nutzen stattdessen decision_function() als Vertrauens-
    # Wert (signierter Abstand zur Entscheidungsgrenze, passend zu predict()).
    margin = float(pipe.decision_function(X)[0])
    label = "ECHTER TREFFER (Treffer)" if pred == 1 else "STOERSIGNAL (Vorbei/Steinschlag/Auffahren)"
    return label, margin


def watch_folder(folder, poll_seconds=1.0):
    """
    Generischer Live-Modus: ueberwacht einen Ordner per Polling und
    klassifiziert jede neue .txt-Datei (Format wie ssValues*.txt).

    Passt zum Fall "der bestehende Logger schreibt weiterhin Dateien".
    Wenn der Sensor stattdessen ueber Seriell (UART) oder SPI direkt am Pi
    angeschlossen wird, muss dieser Teil durch eine passende Leseschleife
    ersetzt werden (siehe EINSATZ_ANLEITUNG.md, Abschnitt 6). Die Funktionen
    extract_features() und predict() bleiben in jedem Fall gleich.
    """
    import time
    print(f"Live-Modus: beobachte Ordner '{folder}' (Strg+C zum Beenden) ...")
    seen = set(os.listdir(folder)) if os.path.isdir(folder) else set()
    while True:
        try:
            current = set(f for f in os.listdir(folder) if f.endswith(".txt"))
        except FileNotFoundError:
            print(f"Ordner '{folder}' existiert nicht (noch) — warte ...")
            time.sleep(poll_seconds)
            continue
        neu = sorted(current - seen)
        for fname in neu:
            path = os.path.join(folder, fname)
            try:
                raw = load_waveform_txt(path)
                label, margin = predict(raw)
                print(f"[{fname}] Vorhersage: {label}  (Entscheidungs-Marge={margin:+.2f})")
            except Exception as e:
                print(f"[{fname}] Fehler beim Verarbeiten: {e}")
        seen = current
        time.sleep(poll_seconds)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung:")
        print("  python3 rpi_inference.py <signaldatei.txt>        # eine Datei klassifizieren")
        print("  python3 rpi_inference.py --live [ordner]          # Ordner ueberwachen (Standard: ./eingang)")
        sys.exit(1)

    if sys.argv[1] == "--live":
        folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_SCRIPT_DIR, "eingang")
        os.makedirs(folder, exist_ok=True)
        watch_folder(folder)
    else:
        raw = load_waveform_txt(sys.argv[1])
        label, margin = predict(raw)
        print(f"Vorhersage: {label}  (Entscheidungs-Marge={margin:+.2f}, je weiter von 0 entfernt, desto sicherer)")
