# ============================================================================
# SCRIPT : Extraction de caractéristiques à partir de données brutes de capteur
# ============================================================================
# Ce script traite les données brutes du capteur, extrait diverses 
# caractéristiques (temporelles, fréquentielles, statistiques) et génère
# un fichier CSV enrichi pour l'entraînement des modèles.
# ============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

# ÉTAPE 1 : Chargement et préparation des données
# Lecture du fichier CSV contenant les données brutes
df = pd.read_csv("sensor_data_raw.csv")

# Extraction des colonnes d'échantillons (colonnes s0, s1, s2, ...)
sample_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
# Tri des colonnes d'échantillons par ordre numérique
sample_cols = sorted(sample_cols, key=lambda c: int(c[1:]))

# Conversion en tableau numpy (shape: nb_signaux x nb_echantillons_par_signal)
X = df[sample_cols].values.astype(float)
N = X.shape[1]  # Nombre d'échantillons par signal

# ÉTAPE 2 : Normalisation (centrage du signal autour du zéro)
# Le capteur ADC a une valeur de repos de 128.0 (0x80)
# On centre tous les signaux autour de zéro pour analyse
BASELINE = 128.0  # Valeur de repos du convertisseur ADC
centered = X - BASELINE  # Signal centré (soustraction du baseline)

# ÉTAPE 3 : Extraction des caractéristiques pour chaque signal
# Une liste pour stocker toutes les caractéristiques extraites
feat_rows = []

# Boucle sur chaque signal (chaque ligne/enregistrement)
for i in range(X.shape[0]):
    raw = X[i]  # Signal brut (non-centré)
    c = centered[i]  # Signal centré autour du zéro
    
    # --- Caractéristiques temporelles du signal ---
    # Position et amplitude du pic maximum (absolu)
    peak_idx = int(np.argmax(np.abs(c)))
    peak_val = c[peak_idx]
    
    # Comptage des zéro-crossings (passages par zéro)
    # Utilise le signe du signal : changement de signe = zéro-crossing
    signs = np.sign(c)
    signs[signs == 0] = 1  # Remplace 0 par 1 pour éviter les problèmes
    zero_crossings = int(np.sum(signs[:-1] != signs[1:]))
    
    # Répartition énergétique : phase d'attaque vs phase de décroissance
    # On divise le signal en deux halves (première moitié vs deuxième moitié)
    half = N // 2
    energy_first = float(np.sum(c[:half] ** 2))  # Énergie première moitié (montée)
    energy_second = float(np.sum(c[half:] ** 2))  # Énergie deuxième moitié (décroissance)
    energy_total = energy_first + energy_second
    # Ratio : quelle proportion de l'énergie dans la phase d'attaque ?
    attack_ratio = energy_first / energy_total if energy_total > 0 else np.nan
    
    # --- Caractéristiques fréquentielles (domaine FFT) ---
    # Calcul de la Transformée de Fourier Rapide (FFT) du signal centré
    spec = np.abs(rfft(c))  # Spectre (amplitude uniquement)
    freqs_bin = np.arange(len(spec))  # Numéros de bins fréquentiels
    spec_sum = spec.sum()
    
    # Centroïde spectral : "centre de masse" du spectre fréquentiel
    spectral_centroid = float((freqs_bin * spec).sum() / spec_sum) if spec_sum > 0 else np.nan
    
    # Bin dominant (fréquence prédominante, en ignorant la composante DC)
    dominant_bin = int(np.argmax(spec[1:]) + 1)  # Saute bin 0 (DC)
    
    # Répartition énergétique fréquentielle : bas/moyen/haut
    # Divise le spectre en 3 zones égales
    n3 = len(spec) // 3
    energy_low = float(np.sum(spec[:n3] ** 2))      # Basses fréquences
    energy_mid = float(np.sum(spec[n3:2*n3] ** 2))  # Fréquences moyennes
    energy_high = float(np.sum(spec[2*n3:] ** 2))   # Hautes fréquences
    spec_energy_total = energy_low + energy_mid + energy_high
    
    # Création du dictionnaire de caractéristiques pour ce signal
    # Chaque clé = caractéristique, chaque valeur = valeur numérique
    feat_rows.append(dict(
        # --- Caractéristiques du signal brut ---
        mean_raw=float(np.mean(raw)),           # Moyenne brute
        std_raw=float(np.std(raw)),             # Écart-type brut
        min_raw=float(np.min(raw)),             # Minimum brut
        max_raw=float(np.max(raw)),             # Maximum brut
        ptp_raw=float(np.ptp(raw)),             # Plage (peak-to-peak) brute
        
        # --- Caractéristiques du signal centré ---
        rms_centered=float(np.sqrt(np.mean(c ** 2))),  # RMS (Racine de la Moyenne des Carrés)
        energy_centered=float(np.sum(c ** 2)),         # Énergie totale du signal
        
        # --- Caractéristiques du pic ---
        peak_abs_amplitude=float(np.abs(peak_val)),    # Amplitude absolue du pic
        peak_signed_amplitude=float(peak_val),         # Amplitude signée du pic (positif/négatif)
        peak_position_norm=peak_idx / N,               # Position normalisée du pic (0.0 à 1.0)
        
        # --- Caractéristiques de zéro-crossing ---
        zero_crossing_rate=zero_crossings / N,         # Taux de zéro-crossing normalisé
        
        # --- Caractéristiques statistiques ---
        skewness=float(stats.skew(c)),                 # Asymétrie de la distribution (skewness)
        kurtosis=float(stats.kurtosis(c)),             # Kurtosis (pointu ou aplati)
        
        # --- Caractéristiques d'énergie temporelle ---
        attack_energy_ratio=attack_ratio,              # Ratio énergétique attaque/total
        
        # --- Caractéristiques fréquentielles ---
        spectral_centroid_bin=spectral_centroid,       # Centroïde spectral (en bins)
        dominant_freq_bin=dominant_bin,                # Fréquence dominante (bin)
        spec_energy_low_frac=energy_low / spec_energy_total if spec_energy_total > 0 else np.nan,    # Fraction d'énergie basses fréq.
        spec_energy_mid_frac=energy_mid / spec_energy_total if spec_energy_total > 0 else np.nan,    # Fraction d'énergie moyennes fréq.
        spec_energy_high_frac=energy_high / spec_energy_total if spec_energy_total > 0 else np.nan,  # Fraction d'énergie hautes fréq.
        
        # --- Caractéristiques d'impacticité ---
        crest_factor=float(np.abs(peak_val) / (np.sqrt(np.mean(c ** 2)) + 1e-9)),  # Facteur de crête (pic / RMS)
    ))

# ÉTAPE 4 : Création du DataFrame de caractéristiques et combinaison avec métadonnées
# Conversion de la liste de dictionnaires en DataFrame Pandas
feat_df = pd.DataFrame(feat_rows)

# Récupération des colonnes de métadonnées (outcome, weapon, material, etc.)
meta_cols = [c for c in df.columns if c not in sample_cols]

# Combinaison des métadonnées avec les caractéristiques extraites
out = pd.concat([df[meta_cols].reset_index(drop=True), feat_df], axis=1)

# ÉTAPE 5 : Sauvegarde et affichage des résultats
# Sauvegarde du DataFrame enrichi dans un nouveau fichier CSV
out.to_csv("sensor_data_features.csv", index=False)

# Affichage des statistiques et aperçus des données
print(out.shape)
print(out.head(10).to_string())
print()
print("Outcome distribution:")
print(out['outcome'].value_counts())
print()
print("Weapon distribution:")
print(out['weapon'].value_counts(dropna=False))
print()
print("Material distribution:")
print(out['material'].value_counts(dropna=False))
