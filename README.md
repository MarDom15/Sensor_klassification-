# 🎯 Gaushorn — Classification tir réel vs événement parasite

## 📌 Objectif

À partir des données brutes fournies par la team sensor (capteur de choc/vibration
sur cible), construire un modèle de classification binaire capable de dire, pour
chaque déclenchement du capteur :

- **Treffer (1)** — un vrai impact de tir sur la cible
- **Parasite (0)** — un déclenchement du capteur qui n'est pas un tir : `Vorbei`
  (tir manqué), `Steinschlag` (impact de pierre/éclat), `Auffahren` (véhicule qui
  approche)

Objectif final : un modèle déployable sur Raspberry Pi pour filtrer les fausses
alertes en temps réel.

## 📂 Arborescence

```
gaushorn_shot_classifier/
├── data/
│   ├── raw/          # données brutes originales (txt + png), telles que reçues
│   └── processed/    # CSV nettoyés (large format + features)
├── src/              # pipeline complet, à exécuter dans l'ordre (01 -> 09)
├── models/           # modèle final entraîné (.joblib)
├── results/          # métriques de comparaison de modèles (CSV)
├── figures/          # graphiques (PNG)
├── deployment/        # script d'inférence autonome pour Raspberry Pi
└── docs/             # présentation du projet
```

## 💾 Données

74 événements capturés (74 fichiers .txt), chacun = une forme d'onde de 512
échantillons (octets 0x00-0xFF, capteur centré ~128). Organisés en 13 séances de
tir (armes P8/G36, distances 10-250m, matériaux Holz/Kunststoff/PE) + 3 fichiers
« Auffahren » hors séance.

> Note : la team a mentionné « 500 informations » — il s'agit des ~512
> échantillons par capture, pas du nombre d'événements (qui est 74).

## 🔧 Pipeline (`src/`)

| Script | Rôle |
|---|---|
| `01_build_raw_csv.py` | Parse tous les .txt, extrait métadonnées (arme, distance, matériau, résultat) + 512 échantillons → `sensor_data_raw.csv` |
| `02_extract_features.py` | Calcule 20 features temporelles/spectrales par événement → `sensor_data_features.csv` |
| `03_plot_waveforms.py` | Visualise les formes d'onde brutes par classe |
| `04_plot_feature_separability.py` | Boxplots + PCA pour vérifier la séparabilité des features |
| `05_train_compare_models_LOGOCV.py` | Entraîne et compare 6 modèles (LOGO-CV par séance de tir) |
| `06_breakdown_by_subclass.py` | Détail des performances par sous-classe (Vorbei/Steinschlag/Auffahren) |
| `07_plot_model_comparison.py` | Graphiques de comparaison des modèles |
| `08_feature_selection_hyperparam_tuning.py` | Réduction de features (20→7) + tuning C/gamma du SVM |
| `09_train_final_model_export.py` | Entraîne le modèle final sur toutes les données et l'exporte |

## ✅ Méthodologie de validation

**Leave-One-Group-Out Cross-Validation (LOGO-CV)**, groupes = séance de tir
(`shot_id`). Chaque tir génère jusqu'à 6 fichiers très corrélés (3 capteurs ×
2 répétitions sur le même événement physique) — une CV classique mélangerait
ces fichiers entre train/test et donnerait des scores artificiellement optimistes.
Le LOGO-CV garantit qu'un tir entier est toujours testé sur un modèle qui ne l'a
jamais vu.

## 🏆 Résultat final

Modèle retenu : **SVM (noyau RBF, C=0.1, gamma=scale)** sur 7 features
sélectionnées (`peak_abs_amplitude`, `energy_centered`, `rms_centered`,
`zero_crossing_rate`, `spectral_centroid_bin`, `crest_factor`,
`attack_energy_ratio`).

En LOGO-CV : accuracy 0.92, balanced accuracy 0.76, F1 (classe parasite) 0.63.

## ⚠️ Limite principale : Steinschlag

Les impacts de pierre (Steinschlag) ont une signature énergétique quasi
identique à un vrai tir (même amplitude, même contenu spectral) — **aucun des
6 modèles testés n'a détecté un seul des 4 exemples Steinschlag** en LOGO-CV.
En excluant ces 4 exemples de l'entraînement, le même modèle atteint 0.90 de
balanced accuracy et 80% de recall sur les parasites restants (Vorbei +
Auffahren) : la limite vient des données (4 exemples, physiquement proches
d'un vrai tir), pas du modèle. Voir `docs/` pour le détail.

## 🚀 Déploiement Raspberry Pi

Voir `deployment/rpi_inference.py` — script autonome (numpy + scikit-learn +
joblib) qui recalcule les 7 features à partir d'une forme d'onde brute et
charge `models/shot_vs_parasite_svm.joblib`. Installation rapide sur
Raspberry Pi OS via les wheels précompilés [piwheels](https://www.piwheels.org/project/scikit-learn/)
(pas de compilation depuis les sources).
