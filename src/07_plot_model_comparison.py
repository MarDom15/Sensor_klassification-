# ============================================================================
# SCRIPT : Visualisation comparative des modèles
# ============================================================================
# Ce script génère deux visualisations :
# 1. Diagrammes en barres : comparaison des scores de tous les modèles
# 2. Heatmap : taux de bonne classification (recall) par sous-classe et modèle
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chargement des résultats de comparaison des modèles
res = pd.read_csv("model_comparison_LOGO_CV.csv")
res = res.sort_values("balanced_accuracy", ascending=False)

# ============================================================================
# FIGURE 1 : Diagrammes en barres comparatifs
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(res))
w = 0.25  # Largeur des barres

# Trace 3 métriques pour chaque modèle
ax.bar(x - w, res["accuracy"], width=w, label="Accuracy (trompeur, classe déséq.)", color="#90a4ae")
ax.bar(x, res["balanced_accuracy"], width=w, label="Balanced Accuracy", color="#1565c0")
ax.bar(x + w, res["f1_parasite"], width=w, label="F1 (classe parasite)", color="#c62828")

# Personnalisation
ax.set_xticks(x)
ax.set_xticklabels(res["model"], rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("score")
ax.set_title("Comparaison des modèles - LOGO-CV (par séance de tir)\nCible binaire: tir réel (Treffer) vs parasite (Vorbei/Steinschlag/Auffahren)")
ax.legend()

# Ajoute une ligne de baseline (classifieur naïf qui prédit toujours "tir")
ax.axhline(65/74, color="gray", linestyle="--", linewidth=1)
ax.text(len(res)-0.5, 65/74+0.02, "baseline (tout prédire 'tir')", fontsize=8, color="gray", ha="right")

fig.tight_layout()
fig.savefig("model_comparison_barplot.png", dpi=130)
print("saved model_comparison_barplot.png")

# ============================================================================
# FIGURE 2 : Heatmap du recall par sous-classe
# ============================================================================
# Chargement du tableau recall par sous-classe
sub = pd.read_csv("recall_by_subclass_per_model.csv", index_col=0)
n_col = sub.pop("n")  # Récupère la colonne avec les effectifs

fig2, ax2 = plt.subplots(figsize=(9, 4.5))

# Crée une heatmap (couleur gradient : vert (bon) à rouge (mauvais))
im = ax2.imshow(sub.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

# Axes : noms des modèles en x, classes en y
ax2.set_xticks(range(len(sub.columns)))
ax2.set_xticklabels(sub.columns, rotation=20, ha="right")
ax2.set_yticks(range(len(sub.index)))
ax2.set_yticklabels([f"{idx} (n={n_col[idx]})" for idx in sub.index])

# Ajoute les valeurs numériques dans chaque cellule
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        v = sub.values[i, j]
        # Texte blanc si fond sombre, noir sinon
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                  color="white" if (v < 0.3 or v > 0.75) else "black", fontsize=10)

ax2.set_title("Taux de bonne classification par sous-classe (LOGO-CV out-of-fold)")

# Ajoute une barre de couleur (légende)
fig2.colorbar(im, ax=ax2, label="taux de bonne classification")
fig2.tight_layout()
fig2.savefig("recall_by_subclass_heatmap.png", dpi=130)
print("saved recall_by_subclass_heatmap.png")
