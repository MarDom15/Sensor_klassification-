# ============================================================================
# SCRIPT : Sélection de features et ajustement des hyperparamètres
# ============================================================================
# Ce script effectue une analyse d'optimisation en 4 étapes :
# 1. Comparaison : 20 features vs 7 features réduites
# 2. Grille de recherche : SVM(RBF) avec différents C et gamma
# 3. Grille de recherche : LogisticRegression avec différents C
# 4. Diagnostic : test sans Steinschlag pour analyser les parasites
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score, confusion_matrix)
import itertools

# Chargement des données
df = pd.read_csv("sensor_data_features.csv")
df["is_shot"] = (df["outcome"] == "Treffer").astype(int)

# Définition des deux ensembles de features
# Les 20 features de base extraites
FULL_FEATS = ["mean_raw","std_raw","min_raw","max_raw","ptp_raw","rms_centered","energy_centered",
             "peak_abs_amplitude","peak_signed_amplitude","peak_position_norm","zero_crossing_rate",
             "skewness","kurtosis","attack_energy_ratio","spectral_centroid_bin","dominant_freq_bin",
             "spec_energy_low_frac","spec_energy_mid_frac","spec_energy_high_frac","crest_factor"]

# Sous-ensemble réduit de 7 features les plus discriminantes
REDUCED_FEATS = ["peak_abs_amplitude", "energy_centered", "rms_centered", "zero_crossing_rate",
                  "spectral_centroid_bin", "crest_factor", "attack_energy_ratio"]

# Préparation des données et validation
y = df["is_shot"].values
groups = df["shot_id"].astype(str).values
logo = LeaveOneGroupOut()

def eval_model(pipe, X, y, groups):
    """
    Évalue un modèle avec validation LOGO-CV.
    
    Arguments:
        pipe : Pipeline sklearn à évaluer
        X : Données d'entraînement
        y : Étiquettes (0=parasite, 1=tir)
        groups : Groupes pour LOGO (shot_id)
    
    Retourne:
        dict : Dictionnaire de métriques d'évaluation
    """
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)
    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        oof_pred[te] = pipe.predict(X[te])
        if hasattr(pipe, "predict_proba"):
            oof_proba[te] = pipe.predict_proba(X[te])[:, 1]
    acc = accuracy_score(y, oof_pred)
    bal = balanced_accuracy_score(y, oof_pred)
    f1p = f1_score(y, oof_pred, pos_label=0, zero_division=0)
    recp = recall_score(y, oof_pred, pos_label=0, zero_division=0)
    precp = precision_score(y, oof_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(y, oof_proba)
    except Exception:
        auc = np.nan
    return dict(accuracy=acc, balanced_accuracy=bal, precision_parasite=precp,
                recall_parasite=recp, f1_parasite=f1p, roc_auc=auc)

# ============================================================================
# ÉTAPE 1 : Comparaison 20 features vs 7 features réduites
# ============================================================================
print("=== 1) Feature set comparison (features 20 vs reduit 7) ===\n")
for feat_name, feats in [("FULL (20 feat)", FULL_FEATS), ("REDUCED (7 feat)", REDUCED_FEATS)]:
    X = df[feats].values
    for model_name, pipe in [
        ("SVM(RBF) C=1", Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=0))])),
        ("LogReg C=1", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=0))])),
    ]:
        r = eval_model(pipe, X, y, groups)
        print(f"{feat_name:18s} | {model_name:15s} | acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
              f"f1_parasite={r['f1_parasite']:.3f} recall_parasite={r['recall_parasite']:.3f} auc={r['roc_auc']:.3f}")

# ============================================================================
# ÉTAPE 2 : Grille de recherche SVM(RBF) sur les features réduites
# ============================================================================
# Teste différentes combinaisons de paramètres C (régularisation) et gamma (kernel)
print("\n=== 2) Grille SVM(RBF) sur features réduites (C x gamma) ===\n")
X = df[REDUCED_FEATS].values
best = None
grid_rows = []
for C in [0.1, 0.5, 1, 2, 5, 10]:
    for gamma in ["scale", "auto", 0.01, 0.05, 0.1, 0.5]:
        pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced", random_state=0))])
        r = eval_model(pipe, X, y, groups)
        r.update(C=C, gamma=gamma)
        grid_rows.append(r)

grid_df = pd.DataFrame(grid_rows).sort_values(["f1_parasite","balanced_accuracy"], ascending=False)
pd.set_option("display.width", 160)
print(grid_df.head(15).to_string(index=False))
grid_df.to_csv("svm_grid_search_LOGO.csv", index=False)

# ============================================================================
# ÉTAPE 3 : Grille de recherche LogisticRegression (paramètre C)
# ============================================================================
# C contrôle le compromis entre régularisation et ajustement aux données
print("\n=== 3) Grille LogisticRegression (C) sur features réduites ===\n")
lr_rows = []
for C in [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10]:
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=0))])
    r = eval_model(pipe, X, y, groups)
    r.update(C=C)
    lr_rows.append(r)
lr_df = pd.DataFrame(lr_rows).sort_values(["f1_parasite","balanced_accuracy"], ascending=False)
print(lr_df.to_string(index=False))
lr_df.to_csv("logreg_grid_search_LOGO.csv", index=False)

# ============================================================================
# ÉTAPE 4 : Diagnostic - Exclusion de "Steinschlag" de l'entraînement
# ============================================================================
# Teste si le modèle peut bien distinguer Vorbei (raté) et Auffahren (approche)
# sans l'aide de Steinschlag (qui peut être très distinct)
print("\n=== 4) Variante: exclure Steinschlag de l'entrainement (diagnostic) ===\n")
mask = df["outcome"] != "Steinschlag"
X_ex = df.loc[mask, REDUCED_FEATS].values
y_ex = y[mask.values]
groups_ex = groups[mask.values]
pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=0))])
r = eval_model(pipe, X_ex, y_ex, groups_ex)
print(f"SVM(RBF) SANS Steinschlag (Vorbei+Auffahren=parasite, n={len(y_ex)}): {r}")
