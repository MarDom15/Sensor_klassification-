# ============================================================================
# SCRIPT : Visualisation des formes d'onde brutes par classe
# ============================================================================
# Ce script charge les données brutes et génère deux visualisations :
# 1. Quelques exemples représentatifs de chaque classe (formes distinctes)
# 2. Toutes les waveforms superposées pour voir la variabilité/dispersion
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sans interface graphique (sauvegarde fichiers)
import matplotlib.pyplot as plt

# Chargement des données brutes (512 échantillons par acquisition)
df = pd.read_csv("sensor_data_raw.csv")

# Extraction des colonnes d'échantillons (s0, s1, ..., s511)
sample_cols = sorted([c for c in df.columns if c.startswith("s") and c[1:].isdigit()], key=lambda c: int(c[1:]))

# Définition des 4 classes principales
classes = ["Treffer", "Vorbei", "Steinschlag", "Auffahren"]
colors = {"Treffer": "#2e7d32", "Vorbei": "#e65100", "Steinschlag": "#6a1b9a", "Auffahren": "#1565c0"}

# ============================================================================
# FIGURE 1 : Exemples représentatifs de chaque classe
# ============================================================================
fig, axes = plt.subplots(len(classes), 1, figsize=(11, 10), sharex=True)
rng = np.random.default_rng(42)  # Seed pour reproductibilité

for ax, cls in zip(axes, classes):
    sub = df[df["outcome"] == cls]
    n_show = min(6, len(sub))  # Affiche max 6 exemples par classe
    idxs = rng.choice(sub.index, size=n_show, replace=False)
    for i in idxs:
        vals = sub.loc[i, sample_cols].values.astype(float) - 128.0  # Centre autour de zéro
        ax.plot(vals, alpha=0.7, linewidth=1)
    ax.set_title(f"{cls}  (n={len(sub)} events total, {n_show} affichés)", loc="left", fontsize=10, color=colors[cls])
    ax.axhline(0, color="gray", linewidth=0.5)  # Ligne de référence
    ax.set_ylabel("amplitude\n(centré, val-128)")

axes[-1].set_xlabel("échantillon (0-511)")
fig.suptitle("Formes d'onde brutes par classe (exemples)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("waveforms_by_class.png", dpi=130)
print("saved waveforms_by_class.png")

# ============================================================================
# FIGURE 2 : Toutes les waveforms superposées par classe
# ============================================================================
# Cette vue permet de voir la variabilité et la dispersion au sein de chaque classe
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
for ax, cls in zip(axes2.flat, classes):
    sub = df[df["outcome"] == cls]
    for i in sub.index:
        vals = sub.loc[i, sample_cols].values.astype(float) - 128.0
        ax.plot(vals, alpha=0.35, linewidth=0.8, color=colors[cls])  # Transparence pour voir chevauchements
    ax.set_title(f"{cls} (n={len(sub)})", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5)
fig2.suptitle("Toutes les formes d'onde superposées, par classe", fontsize=13)
fig2.text(0.5, 0.02, "échantillon (0-511)", ha="center")
fig2.text(0.02, 0.5, "amplitude centrée", va="center", rotation="vertical")
fig2.tight_layout(rect=[0.03, 0.04, 1, 0.96])
fig2.savefig("waveforms_overlay_by_class.png", dpi=130)
print("saved waveforms_overlay_by_class.png")
