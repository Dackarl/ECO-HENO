import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="EcoHeno", layout="wide")

st.title("EcoHeno 1.0 — Predicción de producción de heno")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ecoheno.pkl")

modelo = cargar_modelo()

st.subheader("Ingresar variables del ciclo productivo")

prod_corte = st.number_input("Producción en corte", value=7000)
dia_final = st.slider("Día final de empaque", 1, 6, 3)
sector = st.selectbox("Sector", [1,2,3,4,5,6,7,8])
mes = st.slider("Mes", 1, 12, 6)

if st.button("Predecir producción final"):

    datos = pd.DataFrame({
        "PROD_CORTE":[prod_corte],
        "DIA_FINAL":[dia_final],
        "SECTOR":[sector],
        "MES":[mes]
    })

    pred = modelo.predict(datos)[0]

    st.success(f"Producción final estimada: {pred:,.2f}")

