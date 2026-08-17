# ============================================================================
# SCRIPT : Construction du fichier CSV brut à partir des données de tir
# ============================================================================
# Ce script parse les données brutes de capteur depuis une hiérarchie de dossiers,
# extrait les métadonnées (arme, distance, matériel, résultat) et génère
# un fichier CSV avec toutes les acquisitions brutes (512 échantillons chacune).
# ============================================================================

import os, re, glob
import numpy as np
import pandas as pd
from scipy import stats

# Chemin racine contenant l'arborescence complète des enregistrements
ROOT = "/sessions/pensive-serene-mccarthy/mnt/outputs/work/extracted/190925 gaushorn"

def parse_folder(name):
    """
    Extrait les métadonnées du nom du dossier (arme, distance, matériel, résultat, flag 'eigene').
    
    Exemple : "001_p8 10m Holz eigene" -> weapon="P8", dist=10, material="Holz", outcome="Treffer", eigene=True
    
    Arguments:
        name (str) : Nom du dossier
    
    Retourne:
        tuple : (weapon, distance_m, material, outcome, eigene_flag)
    """
    weapon = None
    tokens = re.split(r'[\s_]+', name)
    for tok in tokens:
        if tok.upper() in ('P8', 'G36'):
            weapon = tok.upper()
            break
    dist = None
    m = re.search(r'(\d+)\s*m\b', name)
    if m: dist = int(m.group(1))
    outcome = "Treffer"  # Valeur par défaut = un vrai tir (hit)
    if re.search(r'vorbei', name, re.IGNORECASE): outcome = "Vorbei"  # Tir raté (miss)
    if re.search(r'steinschlag', name, re.IGNORECASE): outcome = "Steinschlag"  # Rebond de pierre
    material = None
    mats = []
    if re.search(r'holz', name, re.IGNORECASE): mats.append("Holz")  # Bois
    if re.search(r'kunststoff', name, re.IGNORECASE): mats.append("Kunststoff")  # Plastique
    if re.search(r'\bPE\b', name): mats.append("PE")  # Polyéthylène
    if mats: material = "_".join(mats)
    eigene = bool(re.search(r'eigene', name, re.IGNORECASE))  # Flag "propre" (ses propres tests)
    return weapon, dist, material, outcome, eigene

def parse_filename(fname):
    """
    Extrait la position du capteur et le numéro de répétition du tir du nom du fichier.
    
    Exemple : "oben1.txt" -> position="oben" (haut), trigger=1
    
    Arguments:
        fname (str) : Nom du fichier
    
    Retourne:
        tuple : (position, trigger_repeat_number)
    """
    base = fname.replace(".txt","")
    position = None
    m = re.match(r'(oben|mitte|unten)(\d)', base, re.IGNORECASE)
    trigger = None
    if m:
        position = m.group(1).lower()  # "oben" (haut), "mitte" (milieu), "unten" (bas)
        trigger = int(m.group(2))
    else:
        m2 = re.search(r'(\d+)$', base)
        if m2: trigger = int(m2.group(1))
    return position, trigger

def read_waveform(path):
    """
    Lit une waveform brute depuis un fichier texte.
    Le format attendu : ligne 0 = en-tête, ligne 1 = valeurs hexadécimales séparées par des espaces.
    
    Arguments:
        path (str) : Chemin du fichier
    
    Retourne:
        list : Liste de valeurs ADC (entiers 0-255)
    """
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    data_line = lines[1] if len(lines) > 1 else ""
    toks = data_line.split()
    vals = [int(t, 16) for t in toks]  # Conversion hex -> décimal
    return vals

# ============================================================================
# ÉTAPE 1 : Parcours des dossiers et extraction des métadonnées
# ============================================================================
rows_raw = []
rows_feat = []
eid = 0  # ID d'événement unique (incrémenté pour chaque acquisition)

# Parcours les dossiers sous "gaushorn" (chacun = une campagne de tir)
folders = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("__MACOSX"))
for folder in folders:
    fpath = os.path.join(ROOT, folder)
    weapon, dist, material, outcome, eigene = parse_folder(folder)
    shot_id = folder.split("_")[0]
    
    # Parcours tous les fichiers .txt du dossier (= acquisitions pour ce scénario)
    txts = sorted(f for f in os.listdir(fpath) if f.endswith(".txt") and not f.startswith("._"))
    for fname in txts:
        vals = read_waveform(os.path.join(fpath, fname))
        if len(vals) != 512:
            print("WARN unexpected length", folder, fname, len(vals))
        position, trigger = parse_filename(fname)
        eid += 1
        
        # Création du dictionnaire de métadonnées pour cet événement
        meta = dict(
            event_id=eid, shot_id=shot_id, source_folder=folder, filename=fname,
            weapon=weapon, distance_m=dist, material=material, outcome=outcome,
            eigene=eigene, sensor_position=position, trigger_repeat=trigger, variant=None
        )
        # Ajoute les 512 échantillons bruts en colonnes s0, s1, ..., s511
        rows_raw.append({**meta, **{f"s{i}": v for i, v in enumerate(vals)}})

# ============================================================================
# ÉTAPE 2 : Traitement des fichiers isolés (véhicule qui s'approche = "Auffahren")
# ============================================================================
# Fichiers .txt dans le dossier racine (pas de scénario de tir, juste le bruit d'approche)
loose = sorted(f for f in os.listdir(ROOT) if f.endswith(".txt") and not f.startswith("._"))
for fname in loose:
    vals = read_waveform(os.path.join(ROOT, fname))
    if len(vals) != 512:
        print("WARN unexpected length loose", fname, len(vals))
    eid += 1
    variant = "mit Scheibe" if "Scheibe" in fname else None
    meta = dict(
        event_id=eid, shot_id="loose", source_folder="(root)", filename=fname,
        weapon=None, distance_m=None, material=None, outcome="Auffahren",
        eigene=False, sensor_position=None, trigger_repeat=None, variant=variant
    )
    rows_raw.append({**meta, **{f"s{i}": v for i, v in enumerate(vals)}})

# ============================================================================
# ÉTAPE 3 : Création du DataFrame et sauvegarde
# ============================================================================
df_raw = pd.DataFrame(rows_raw)
print(df_raw.shape)
print(df_raw[['event_id','source_folder','filename','weapon','distance_m','material','outcome','eigene','sensor_position','trigger_repeat','variant']])
df_raw.to_csv("/sessions/pensive-serene-mccarthy/mnt/outputs/work/sensor_data_raw.csv", index=False)
