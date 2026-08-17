from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.predict import predict_signal_array
from app.preprocess import infer_context_from_path, read_waveform_from_text

st.set_page_config(page_title="Gaushorn Shot Classifier", page_icon="🚨", layout="wide")

st.title("🚨 Gaushorn Shot Classifier")
st.caption("Classification d'un signal brut : tir réel vs parasite")

with st.sidebar:
    st.header("Informations")
    st.markdown(
        """
        Cette application charge un signal brut issu du capteur,
        calcule les features du modèle et retourne une prédiction.
        Elle affiche aussi le contexte métier détecté à partir du fichier:
        distance, matériau, arme et position du capteur.
        """
    )

uploaded_file = st.file_uploader("Téléverser un fichier texte brut", type=["txt", "csv"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8", errors="replace")

    try:
        signal = read_waveform_from_text(text)
    except Exception as exc:
        st.error(f"Impossible d'interpréter le fichier : {exc}")
        st.stop()

    context = infer_context_from_path(uploaded_file.name)
    result = predict_signal_array(signal)

    st.subheader("Contexte détecté")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Arme", context.get("weapon") or "Inconnue")
    meta_cols[1].metric("Distance", f"{context.get('distance_m')} m" if context.get("distance_m") is not None else "Inconnue")
    meta_cols[2].metric("Matériau", context.get("material") or "Inconnu")
    meta_cols[3].metric("Position", context.get("sensor_position") or "Inconnue")

    st.subheader("Signal brut")
    df_signal = pd.DataFrame({"Index": np.arange(len(signal)), "ADC": signal})
    st.line_chart(df_signal.set_index("Index"))

    st.subheader("Résultat")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prédiction", result["label"])
    col2.metric("Confiance", f"{result['confidence'] * 100:.1f}%")
    col3.metric("Probabilité parasite", f"{result['probabilities']['Parasite'] * 100:.1f}%")

    st.write("Probabilités détaillées :")
    st.json(result["probabilities"])

    st.download_button(
        label="Télécharger les résultats JSON",
        data=str({**result, "context": context}).replace("'", '"'),
        file_name="prediction_result.json",
        mime="application/json",
    )
else:
    st.info("Téléversez un fichier texte brut pour lancer la classification.")

    example_signal = np.array([128.0 + 20.0 * np.sin(i / 20.0) for i in range(512)], dtype=float)
    if st.button("Tester avec un signal de démonstration"):
        result = predict_signal_array(example_signal)
        st.success(f"Prédiction : {result['label']} ({result['confidence'] * 100:.1f}% de confiance)")
