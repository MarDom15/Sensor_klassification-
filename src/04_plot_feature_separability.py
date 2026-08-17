# ============================================================================
# SCRIPT : Analyse de la séparabilité des caractéristiques
# ============================================================================
# Ce script visualise la distribution des features extraites et teste
# la séparation des classes en utilisant l'Analyse en Composantes Principales (PCA).
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Chargement des caractéristiques extraites
df = pd.read_csv("sensor_data_features.csv")

# Création d'une variable binaire : Tir réel (Treffer) vs Parasite (tout le reste)
df["is_shot"] = (df["outcome"] == "Treffer").astype(int)
df["label"] = df["outcome"].map(lambda o: "Tir réel (Treffer)" if o == "Treffer" else f"Parasite ({o})")

# Sélection de 8 features clés pour visualisation en boxplot
feat_cols = ["peak_abs_amplitude", "energy_centered", "rms_centered", "zero_crossing_rate",
             "attack_energy_ratio", "spectral_centroid_bin", "crest_factor", "skewness"]

# Couleurs pour chaque classe
colors = {"Treffer": "#2e7d32", "Vorbei": "#e65100", "Steinschlag": "#6a1b9a", "Auffahren": "#1565c0"}
classes_order = ["Treffer", "Vorbei", "Steinschlag", "Auffahren"]

# ============================================================================
# FIGURE 1 : Boxplots des features par classe (outcome)
# ============================================================================
# Chaque subplot affiche la distribution d'une feature pour les 4 classes
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, feat in zip(axes.flat, feat_cols):
    # Prépare les données pour le boxplot (une liste par classe)
    data = [df.loc[df["outcome"] == c, feat].dropna().values for c in classes_order]
    bp = ax.boxplot(data, labels=classes_order, patch_artist=True, showmeans=True)
    
    # Coulore les boîtes selon la classe
    for patch, c in zip(bp["boxes"], classes_order):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.5)
    
    # Ajoute les points individuels (scatter) pour voir la variabilité réelle
    for c_idx, c in enumerate(classes_order):
        yvals = df.loc[df["outcome"] == c, feat].dropna().values
        xvals = np.random.normal(c_idx + 1, 0.05, size=len(yvals))  # Jitter horizontal
        ax.scatter(xvals, yvals, s=12, color=colors[c], edgecolor="black", linewidth=0.3, zorder=3)
    
    ax.set_title(feat, fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

fig.suptitle("Distribution des features par classe (outcome)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("feature_boxplots_by_class.png", dpi=130)
print("saved feature_boxplots_by_class.png")

# ============================================================================
# FIGURE 2 : Analyse PCA (séparation en 2D)
# ============================================================================
# Récupère toutes les colonnes numériques (toutes les features)
all_feat_cols = [c for c in df.columns if df[c].dtype != object and c not in
                  ("event_id", "distance_m", "trigger_repeat", "is_shot", "eigene")]

# Prépare les données : remplissage des NaN avec la médiane
X = df[all_feat_cols].fillna(df[all_feat_cols].median())

# Normalise les features (standardisation) -> essentiel pour PCA
Xs = StandardScaler().fit_transform(X)

# Applique PCA : projection sur les 2 premières composantes principales
pca = PCA(n_components=2, random_state=0)
Xp = pca.fit_transform(Xs)

# Crée deux subplots : classification 4-clases vs classification binaire
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1 : 4 classes (Treffer, Vorbei, Steinschlag, Auffahren)
for c in classes_order:
    mask = df["outcome"] == c
    ax3[0].scatter(Xp[mask.values, 0], Xp[mask.values, 1], label=c, color=colors[c], s=50, edgecolor="black", linewidth=0.4)
ax3[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")  # % variance expliquée
ax3[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax3[0].set_title("PCA - 4 classes (outcome)")
ax3[0].legend(fontsize=8)

# Subplot 2 : Classification binaire (Tir vs Parasite)
binary_colors = {1: "#2e7d32", 0: "#c62828"}
binary_label = {1: "Tir réel", 0: "Parasite"}
for b in [1, 0]:
    mask = df["is_shot"] == b
    ax3[1].scatter(Xp[mask.values, 0], Xp[mask.values, 1], label=binary_label[b], color=binary_colors[b], s=50, edgecolor="black", linewidth=0.4)
ax3[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax3[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax3[1].set_title("PCA - binaire (tir vs parasite)")
ax3[1].legend(fontsize=8)

fig3.suptitle("Séparabilité des features (projection PCA sur toutes les features)", fontsize=13)
fig3.tight_layout(rect=[0, 0, 1, 0.94])
fig3.savefig("feature_pca_scatter.png", dpi=130)
print("saved feature_pca_scatter.png")
