# ============================================================================
# SCRIPT : Comparaison de plusieurs modèles avec validation Leave-One-Group-Out
# ============================================================================
# Ce script entraîne 6 modèles différents avec LOGO-CV (Leave-One-Group-Out Cross-Validation).
# Validation LOGO : pour chaque séance de tir (shot_id), on entraîne sur les autres séances
# et on teste sur la séance laissée de côté. Cela simule une performance en cas de nouvelle
# séance de tir non vista.
#
# Objectif : Classification BINAIRE = Tir réel (Treffer) vs Parasite (tout le reste)
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score, confusion_matrix)

# Chargement des données avec les caractéristiques extraites
df = pd.read_csv("sensor_data_features.csv")

# Création de la cible binaire : 1 = Treffer (tir réel), 0 = Parasite (autre)
df["is_shot"] = (df["outcome"] == "Treffer").astype(int)

# Sélection de 20 features pour l'entraînement
feat_cols = ["mean_raw","std_raw","min_raw","max_raw","ptp_raw","rms_centered","energy_centered",
             "peak_abs_amplitude","peak_signed_amplitude","peak_position_norm","zero_crossing_rate",
             "skewness","kurtosis","attack_energy_ratio","spectral_centroid_bin","dominant_freq_bin",
             "spec_energy_low_frac","spec_energy_mid_frac","spec_energy_high_frac","crest_factor"]

# Préparation des données
X = df[feat_cols].values
y = df["is_shot"].values
groups = df["shot_id"].astype(str).values

# Affichage des statistiques
print("n_samples:", len(y), "| n_shot(1):", y.sum(), "| n_parasite(0):", (y==0).sum())
print("n_groups:", len(np.unique(groups)))

# ============================================================================
# Définition des 6 modèles à tester
# ============================================================================
models = {
    "LogisticRegression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0))]),
    "KNN(k=5)": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "SVM(RBF)": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0))]),
    "DecisionTree": Pipeline([("sc", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0))]),
    "RandomForest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=0))]),
    "GradientBoosting": Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=0))]),
}

# ============================================================================
# Validation LOGO (Leave-One-Group-Out)
# ============================================================================
logo = LeaveOneGroupOut()
results = []
oof_store = {}  # Stockage des prédictions out-of-fold pour analyse ultérieure

for name, pipe in models.items():
    oof_pred = np.zeros(len(y), dtype=int)      # Prédictions out-of-fold
    oof_proba = np.zeros(len(y), dtype=float)   # Probabilités out-of-fold
    
    # Boucle : pour chaque groupe (séance de tir), utilise les autres pour entraîner
    for train_idx, test_idx in logo.split(X, y, groups):
        pipe.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = pipe.predict(X[test_idx])
        # Récupère les probabilités prédites (utiles pour AUC)
        if hasattr(pipe, "predict_proba"):
            oof_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            oof_proba[test_idx] = oof_pred[test_idx]
    
    # Stockage des résultats pour analyse ultérieure
    oof_store[name] = (oof_pred, oof_proba)

    # Calcul des métriques d'évaluation
    acc = accuracy_score(y, oof_pred)
    bal_acc = balanced_accuracy_score(y, oof_pred)
    
    # Métriques pour la classe "parasite" (0)
    prec0 = precision_score(y, oof_pred, pos_label=0, zero_division=0)
    rec0 = recall_score(y, oof_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y, oof_pred, pos_label=0, zero_division=0)
    
    # Métriques pour la classe "tir" (1)
    prec1 = precision_score(y, oof_pred, pos_label=1, zero_division=0)
    rec1 = recall_score(y, oof_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y, oof_pred, pos_label=1, zero_division=0)
    
    # AUC-ROC (aire sous la courbe)
    try:
        auc = roc_auc_score(y, oof_proba)
    except Exception:
        auc = np.nan
    
    # Matrice de confusion
    cm = confusion_matrix(y, oof_pred, labels=[0,1])
    
    # Stockage des résultats
    results.append(dict(model=name, accuracy=acc, balanced_accuracy=bal_acc,
                         precision_parasite=prec0, recall_parasite=rec0, f1_parasite=f1_0,
                         precision_tir=prec1, recall_tir=rec1, f1_tir=f1_1, roc_auc=auc,
                         tn=cm[0,0], fp=cm[0,1], fn=cm[1,0], tp=cm[1,1]))

# ============================================================================
# Affichage et sauvegarde des résultats
# ============================================================================
res_df = pd.DataFrame(results).sort_values("f1_parasite", ascending=False)
pd.set_option("display.width", 160)
print()
print(res_df.to_string(index=False))
res_df.to_csv("model_comparison_LOGO_CV.csv", index=False)

# Sauvegarde des prédictions out-of-fold (utiles pour analyses détaillées)
import pickle
with open("oof_store.pkl", "wb") as f:
    pickle.dump({"y": y, "groups": groups, "oof_store": oof_store, "df_meta": df[["event_id","source_folder","filename","outcome","shot_id"]]}, f)
print("\nSaved model_comparison_LOGO_CV.csv and oof_store.pkl")
