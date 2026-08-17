# Gaushorn — Echter Treffer oder Störsignal?

## Ziel des Projekts

Die Sensor-Team hat Rohdaten von einem Stoß-Sensor geschickt. Der Sensor sitzt
an einer Schießscheibe. Jedes Mal, wenn der Sensor auslöst, speichert er ein
Signal. Aber nicht jedes Signal ist ein echter Treffer.

Wir bauen ein Modell. Das Modell soll sagen: War das ein echter Treffer, oder
war es ein Störsignal?

- **Treffer (1)** — ein echter Schuss hat die Scheibe getroffen
- **Störsignal (0)** — der Sensor hat ausgelöst, aber es war kein Treffer:
  `Vorbei` (Schuss verfehlt die Scheibe), `Steinschlag` (ein Stein trifft die
  Scheibe), `Auffahren` (ein Fahrzeug kommt näher)

Das fertige Modell soll später auf einem Raspberry Pi laufen. Dort soll es
Fehlalarme direkt vor Ort erkennen.

## Ordnerstruktur

```
gaushorn_shot_classifier/
├── data/
│   ├── raw/          # Original-Rohdaten (txt + png), so wie geliefert
│   └── processed/    # saubere CSV-Dateien (Rohdaten + Merkmale)
├── src/              # alle Skripte, der Reihe nach ausführen (01 bis 09)
├── models/           # das fertige, trainierte Modell (.joblib)
├── results/          # Vergleichstabellen der Modelle (CSV)
├── figures/          # alle Grafiken (PNG)
├── deployment/        # eigenständiges Skript für den Raspberry Pi
└── docs/             # Präsentation des Projekts
```

## Die Daten

74 Ereignisse wurden aufgenommen (74 .txt-Dateien). Jede Datei ist ein Signal
mit 512 Messpunkten (Werte von 0x00 bis 0xFF, Mittelwert ~128). Die Daten
kommen aus 13 Schuss-Serien (Waffen P8/G36, Entfernung 10–250 m, Material
Holz/Kunststoff/PE) plus 3 zusätzliche Dateien "Auffahren".

> Hinweis: Die Team sagte "500 Informationen". Das sind die ~512 Messpunkte
> in jeder Aufnahme — nicht die Anzahl der Ereignisse. Es gibt wirklich 74
> Ereignisse in den Daten.

## Die Programme (`src/`)

| Skript | Was es macht |
|---|---|
| `01_build_raw_csv.py` | Liest alle .txt-Dateien. Holt Waffe, Entfernung, Material, Ergebnis und die 512 Messwerte heraus → `sensor_data_raw.csv` |
| `02_extract_features.py` | Berechnet 20 Merkmale pro Ereignis → `sensor_data_features.csv` |
| `03_plot_waveforms.py` | Zeigt die rohen Signale, sortiert nach Klasse |
| `04_plot_feature_separability.py` | Boxplots und PCA — zeigen, wie gut die Merkmale trennen |
| `05_train_compare_models_LOGOCV.py` | Trainiert und vergleicht 6 Modelle (LOGO-CV, gruppiert nach Schuss-Serie) |
| `06_breakdown_by_subclass.py` | Zeigt die Ergebnisse pro Unterklasse (Vorbei/Steinschlag/Auffahren) |
| `07_plot_model_comparison.py` | Grafiken zum Modellvergleich |
| `08_feature_selection_hyperparam_tuning.py` | Weniger Merkmale (20→7) und bessere Einstellungen für das SVM |
| `09_train_final_model_export.py` | Trainiert das finale Modell mit allen Daten und speichert es |

## Wie wir richtig testen

**Leave-One-Group-Out Cross-Validation (LOGO-CV)**, Gruppe = eine Schuss-Serie
(`shot_id`). Ein Schuss erzeugt bis zu 6 Dateien. Diese Dateien sind sich sehr
ähnlich (3 Sensoren × 2 Wiederholungen, gleicher Treffer). Eine normale
Kreuzvalidierung würde diese Dateien mischen — das Ergebnis wäre zu gut, aber
nicht echt. LOGO-CV testet immer eine ganze Schuss-Serie, die das Modell noch
nie gesehen hat.

## Das Ergebnis

Bestes Modell: **SVM mit RBF-Kernel** (C=0.1, gamma=scale), mit 7 Merkmalen
(`peak_abs_amplitude`, `energy_centered`, `rms_centered`,
`zero_crossing_rate`, `spectral_centroid_bin`, `crest_factor`,
`attack_energy_ratio`).

Ergebnis in LOGO-CV: Accuracy 0.92, Balanced Accuracy 0.76, F1 (Störsignal-
Klasse) 0.63.

## Die größte Schwäche: Steinschlag

Ein Steinschlag sieht dem Sensor fast genauso aus wie ein echter Treffer —
gleiche Energie, gleiches Frequenzmuster. **Keines der 6 getesteten Modelle
hat auch nur einen der 4 Steinschlag-Fälle richtig erkannt.** Ohne diese 4
Fälle im Training steigt die Balanced Accuracy auf 0.90, und 80% der
restlichen Störsignale (Vorbei + Auffahren) werden erkannt. Das zeigt: Das
Problem liegt an den Daten (nur 4 Beispiele, sehr ähnlich zu einem echten
Treffer) — nicht am Modell. Mehr dazu in `docs/`.

## Einsatz auf dem Raspberry Pi

Siehe `deployment/rpi_inference.py` — ein eigenständiges Skript (numpy +
scikit-learn + joblib). Es berechnet die 7 Merkmale direkt aus dem rohen
Signal und lädt `models/shot_vs_parasite_svm.joblib`. Die Installation auf
Raspberry Pi OS geht schnell, weil fertige Pakete von
[piwheels](https://www.piwheels.org/project/scikit-learn/) genutzt werden
(kein langes Kompilieren nötig).
