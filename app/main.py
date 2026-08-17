from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import streamlit as st

from app.predict import predict_signal_array
from app.preprocess import infer_context_from_path, read_waveform_from_text

st.set_page_config(page_title="Gaushorn Shot Classifier", page_icon="🚨", layout="wide")

st.title("🚨 Gaushorn Shot Classifier")
st.caption("Classification d'un signal brut : tir réel vs parasite")

with st.sidebar:
    st.header("Contexte de tir")
    st.markdown(
        """
        Saisissez les paramètres métier du tir pour améliorer le contexte d’analyse.
        Les champs sont exportés avec le résultat pour la traçabilité.
        """
    )

    default_context = {"weapon": "P8", "distance_m": 10, "material": "Holz", "sensor_position": "mitte"}
    detected = {}

    uploaded_file = st.file_uploader("Téléverser un fichier texte brut", type=["txt", "csv"])
    if uploaded_file is not None:
        detected = infer_context_from_path(uploaded_file.name)
        default_context = {**default_context, **{key: value for key, value in detected.items() if value is not None}}

    weapon = st.selectbox("Arme", options=["P8", "G36", "Inconnue"], index=["P8", "G36", "Inconnue"].index(default_context.get("weapon", "Inconnue")))
    distance = st.number_input("Distance (m)", min_value=0, max_value=500, value=int(default_context.get("distance_m") or 10), step=1)
    material = st.selectbox("Matériau / type de cible", options=["Holz", "Kunststoff", "PE", "Inconnu"], index=["Holz", "Kunststoff", "PE", "Inconnu"].index(default_context.get("material", "Inconnu")))
    sensor_position = st.selectbox("Position du capteur", options=["oben", "mitte", "unten", "Inconnue"], index=["oben", "mitte", "unten", "Inconnue"].index(default_context.get("sensor_position", "Inconnue")))
    temperature = st.number_input("Température (°C)", min_value=-40.0, max_value=80.0, value=20.0, step=0.5)
    shot_date = st.date_input("Date", value=date.today())
    shot_time = st.time_input("Heure", value=datetime.now().time())

    st.markdown("---")
    st.caption("Le contexte métier est ajouté au JSON de sortie pour archivage et audit.")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8", errors="replace")

    try:
        signal = read_waveform_from_text(text)
    except Exception as exc:
        st.error(f"Impossible d'interpréter le fichier : {exc}")
        st.stop()

    context = {
        "weapon": weapon,
        "distance_m": int(distance),
        "material": material,
        "sensor_position": sensor_position,
        "temperature_c": float(temperature),
        "date": shot_date.isoformat(),
        "time": shot_time.strftime("%H:%M:%S"),
        "source_folder": detected.get("source_folder") if "detected" in locals() else None,
        "filename": uploaded_file.name,
    }
    result = predict_signal_array(signal)

    st.subheader("Contexte de tir")
    meta_cols = st.columns(5)
    meta_cols[0].metric("Arme", context["weapon"])
    meta_cols[1].metric("Distance", f"{context['distance_m']} m")
    meta_cols[2].metric("Matériau", context["material"])
    meta_cols[3].metric("Position", context["sensor_position"])
    meta_cols[4].metric("Température", f"{context['temperature_c']} °C")

    st.info(f"Date: {context['date']} | Heure: {context['time']}")

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
