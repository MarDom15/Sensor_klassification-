# ============================================================================
# SCRIPT : Analyse du recall par sous-classe
# ============================================================================
# Ce script charge les prédictions out-of-fold de tous les modèles et calcule
# le taux de bonne classification (recall) pour CHAQUE sous-classe :
# Treffer (tir réel), Vorbei (raté), Steinschlag (rebond), Auffahren (approche)
# ============================================================================

import pickle
import pandas as pd
import numpy as np

# Chargement des prédictions out-of-fold sauvegardées précédemment
with open("oof_store.pkl", "rb") as f:
    d = pickle.load(f)

y = d["y"]                          # Vraies étiquettes (0 ou 1)
oof_store = d["oof_store"]          # Dict de prédictions par modèle
meta = d["df_meta"].reset_index(drop=True)  # Métadonnées (outcome, etc.)

# ============================================================================
# Calcul du recall par sous-classe
# ============================================================================
rows = []
for name, (pred, proba) in oof_store.items():
    tmp = meta.copy()
    tmp["y_true"] = y
    tmp["y_pred"] = pred
    tmp["correct"] = (tmp["y_true"] == tmp["y_pred"]).astype(int)
    
    # Groupe par outcome et calcule le taux de bonne classification
    grp = tmp.groupby("outcome")["correct"].agg(["mean", "count"]).reset_index()
    grp["model"] = name
    rows.append(grp)

# ============================================================================
# Création d'un tableau pivot : outcomes x modèles
# ============================================================================
out = pd.concat(rows, ignore_index=True)

# Pivot : lignes = outcome, colonnes = modèles, valeurs = recall
pivot = out.pivot(index="outcome", columns="model", values="mean")

# Réordonne les colonnes dans un ordre logique
pivot = pivot[["LogisticRegression","KNN(k=5)","SVM(RBF)","DecisionTree","RandomForest","GradientBoosting"]]

# Réordonne les lignes dans l'ordre des classes
pivot = pivot.reindex(["Treffer","Vorbei","Steinschlag","Auffahren"])

# Ajoute une colonne avec le nombre d'échantillons par classe
counts = out.drop_duplicates("outcome").set_index("outcome")["count"].reindex(["Treffer","Vorbei","Steinschlag","Auffahren"])
pivot.insert(0, "n", counts)

# ============================================================================
# Affichage et sauvegarde
# ============================================================================
pd.set_option("display.width", 160)
print("Taux de bonne classification (recall) PAR SOUS-CLASSE et par modèle (LOGO-CV, out-of-fold):\n")
print(pivot.round(2).to_string())

# Sauvegarde avec 3 décimales pour plus de précision
pivot.round(3).to_csv("recall_by_subclass_per_model.csv")
