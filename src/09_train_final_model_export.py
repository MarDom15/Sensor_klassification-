# ============================================================================
# SCRIPT : Entraînement du modèle final et export
# ============================================================================
# Ce script entraîne le modèle final sélectionné (SVM RBF) sur TOUTES les données,
# puis le sauvegarde dans un fichier joblib pour déploiement ultérieur.
#
# Le modèle utilise 7 features clés et est prêt à être utilisé en production
# pour classifier des acquisitions brutes : Tir réel (1) vs Parasite (0)
# ============================================================================

import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Chargement des données avec les caractéristiques extraites
df = pd.read_csv("sensor_data_features.csv")

# Création de la cible binaire : 1 = Treffer (tir réel), 0 = Parasite
df["is_shot"] = (df["outcome"] == "Treffer").astype(int)

# Sélection des 7 features optimales (déterminées par la grille de recherche)
REDUCED_FEATS = ["peak_abs_amplitude", "energy_centered", "rms_centered", "zero_crossing_rate",
                  "spectral_centroid_bin", "crest_factor", "attack_energy_ratio"]

# Préparation des données d'entraînement
X = df[REDUCED_FEATS].values
y = df["is_shot"].values

# ============================================================================
# Création et entraînement du modèle final
# ============================================================================
# Pipeline : standardisation des features + SVM RBF
# Hyperparamètres sélectionnés après optimisation :
# - C=0.1 (régularisation modérée)
# - gamma="scale" (1 / (n_features * X.var()))
# - probability=True (pour obtenir les probabilités)
final_pipe = Pipeline([
    ("sc", StandardScaler()),  # Normalise chaque feature (moyenne=0, std=1)
    ("clf", SVC(kernel="rbf", C=0.1, gamma="scale", probability=True, class_weight="balanced", random_state=0)),
])

# Entraîne le pipeline sur toutes les données
final_pipe.fit(X, y)

# ============================================================================
# Sauvegarde du modèle
# ============================================================================
# Stocke le pipeline, les noms de features, et la version de sklearn (pour compatibilité)
joblib.dump({"pipeline": final_pipe, "features": REDUCED_FEATS, "sklearn_version": sklearn.__version__},
            "shot_vs_parasite_svm.joblib")
print("Modele final entraine sur", len(y), "echantillons, features:", REDUCED_FEATS)
print("sklearn version utilisee:", sklearn.__version__)

# ============================================================================
# Validation : vérification que le modèle charge et fonctionne correctement
# ============================================================================
# Charge le modèle depuis le fichier et teste la prédiction
loaded = joblib.load("shot_vs_parasite_svm.joblib")
preds = loaded["pipeline"].predict(X)
print("Accuracy sur train (pas de CV, juste sanity check):", (preds == y).mean())
