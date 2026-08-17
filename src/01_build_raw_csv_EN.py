# ============================================================================
# SCRIPT : Building raw CSV file from shooting data
# ============================================================================
# This script parses raw sensor data from a folder hierarchy,
# extracts metadata (weapon, distance, material, outcome) and generates
# a CSV file with all raw acquisitions (512 samples each).
# ============================================================================

import os, re, glob
import numpy as np
import pandas as pd
from scipy import stats

# Root path containing the complete hierarchy of recordings
ROOT = "/sessions/pensive-serene-mccarthy/mnt/outputs/work/extracted/190925 gaushorn"

def parse_folder(name):
    """
    Extracts metadata from folder name (weapon, distance, material, outcome, 'eigene' flag).
    
    Example: "001_p8 10m Holz eigene" -> weapon="P8", dist=10, material="Holz", outcome="Treffer", eigene=True
    
    Arguments:
        name (str) : Folder name
    
    Returns:
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
    outcome = "Treffer"  # Default value = real shot (hit)
    if re.search(r'vorbei', name, re.IGNORECASE): outcome = "Vorbei"  # Missed shot
    if re.search(r'steinschlag', name, re.IGNORECASE): outcome = "Steinschlag"  # Stone ricochet
    material = None
    mats = []
    if re.search(r'holz', name, re.IGNORECASE): mats.append("Holz")  # Wood
    if re.search(r'kunststoff', name, re.IGNORECASE): mats.append("Kunststoff")  # Plastic
    if re.search(r'\bPE\b', name): mats.append("PE")  # Polyethylene
    if mats: material = "_".join(mats)
    eigene = bool(re.search(r'eigene', name, re.IGNORECASE))  # Flag "own" (own tests)
    return weapon, dist, material, outcome, eigene

def parse_filename(fname):
    """
    Extracts sensor position and shot repeat number from filename.
    
    Example: "oben1.txt" -> position="oben" (top), trigger=1
    
    Arguments:
        fname (str) : Filename
    
    Returns:
        tuple : (position, trigger_repeat_number)
    """
    base = fname.replace(".txt","")
    position = None
    m = re.match(r'(oben|mitte|unten)(\d)', base, re.IGNORECASE)
    trigger = None
    if m:
        position = m.group(1).lower()  # "oben" (top), "mitte" (middle), "unten" (bottom)
        trigger = int(m.group(2))
    else:
        m2 = re.search(r'(\d+)$', base)
        if m2: trigger = int(m2.group(1))
    return position, trigger

def read_waveform(path):
    """
    Reads raw waveform from text file.
    Expected format: line 0 = header, line 1 = hexadecimal values separated by spaces.
    
    Arguments:
        path (str) : File path
    
    Returns:
        list : List of ADC values (integers 0-255)
    """
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()
    data_line = lines[1] if len(lines) > 1 else ""
    toks = data_line.split()
    vals = [int(t, 16) for t in toks]  # Convert hex -> decimal
    return vals

# ============================================================================
# STEP 1 : Traverse folders and extract metadata
# ============================================================================
rows_raw = []
rows_feat = []
eid = 0  # Unique event ID (incremented for each acquisition)

# Traverse folders under "gaushorn" (each = one shooting campaign)
folders = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("__MACOSX"))
for folder in folders:
    fpath = os.path.join(ROOT, folder)
    weapon, dist, material, outcome, eigene = parse_folder(folder)
    shot_id = folder.split("_")[0]
    
    # Traverse all .txt files in folder (= acquisitions for this scenario)
    txts = sorted(f for f in os.listdir(fpath) if f.endswith(".txt") and not f.startswith("._"))
    for fname in txts:
        vals = read_waveform(os.path.join(fpath, fname))
        if len(vals) != 512:
            print("WARN unexpected length", folder, fname, len(vals))
        position, trigger = parse_filename(fname)
        eid += 1
        
        # Create metadata dictionary for this event
        meta = dict(
            event_id=eid, shot_id=shot_id, source_folder=folder, filename=fname,
            weapon=weapon, distance_m=dist, material=material, outcome=outcome,
            eigene=eigene, sensor_position=position, trigger_repeat=trigger, variant=None
        )
        # Add 512 raw samples as columns s0, s1, ..., s511
        rows_raw.append({**meta, **{f"s{i}": v for i, v in enumerate(vals)}})

# ============================================================================
# STEP 2 : Process loose files (vehicle approaching = "Auffahren")
# ============================================================================
# .txt files in root folder (no shooting scenario, just approach noise)
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
# STEP 3 : Create DataFrame and save
# ============================================================================
df_raw = pd.DataFrame(rows_raw)
print(df_raw.shape)
print(df_raw[['event_id','source_folder','filename','weapon','distance_m','material','outcome','eigene','sensor_position','trigger_repeat','variant']])
df_raw.to_csv("/sessions/pensive-serene-mccarthy/mnt/outputs/work/sensor_data_raw.csv", index=False)
