import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------

st.set_page_config(
    page_title="EcoHeno 1.0",
    layout="wide"
)

st.title("EcoHeno 1.0 — Predicción de producción de heno")
st.caption("Cuadro maestro de decisión: predicción + simulación + visualización ejecutiva.")

# -----------------------------
# CARGAR MODELO
# -----------------------------

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ecoheno.pkl")

modelo = cargar_modelo()


# -----------------------------
# FUNCIÓN DE PREDICCIÓN
# (AJUSTA EL ORDEN DE VARIABLES SI TU MODELO LO REQUIERE)
# -----------------------------

def predecir_produccion(prod_corte, dia, sector, mes):
    
    X = pd.DataFrame([{
        "Produccion_corte": prod_corte,
        "Dia_Empaque": dia,
        "Sector": sector,
        "Mes": mes
    }])

    pred = modelo.predict(X)[0]
    
    # Nunca dejar negativos (error típico de modelos)
    return max(pred, 0)


# -----------------------------
# SIDEBAR (CUADRO MAESTRO)
# -----------------------------

st.sidebar.header("Cuadro maestro — Variables del ciclo")

prod_corte = st.sidebar.number_input(
    "Producción en corte",
    min_value=0.0,
    value=8000.0,
    step=100.0
)

dia_final = st.sidebar.slider(
    "Día final de empaque",
    1, 6, 3
)

sector = st.sidebar.selectbox(
    "Sector",
    [1,2,3,4,5,6]
)

mes = st.sidebar.slider(
    "Mes",
    1,12,1
)

# -----------------------------
# SIMULACIÓN POR DÍAS
# -----------------------------

def simular_dias(prod_corte, sector, mes):
    
    filas = []
    
    for d in range(1,7):
        pred = predecir_produccion(prod_corte, d, sector, mes)
        
        filas.append({
            "Dia_Empaque": d,
            "Produccion_Estimada": pred
        })
        
    sim = pd.DataFrame(filas)

    # Brecha vs corte (NO usar "pérdida")
    sim["Brecha_vs_corte"] = prod_corte - sim["Produccion_Estimada"]

    # Rendimiento real del proceso
    sim["Rendimiento_%"] = (sim["Produccion_Estimada"] / prod_corte) * 100

    # Cambio vs día 1
    base = sim.loc[sim["Dia_Empaque"] == 1, "Produccion_Estimada"].iloc[0]

    sim["Delta_vs_dia1"] = sim["Produccion_Estimada"] - base
    sim["Cambio_marginal"] = sim["Produccion_Estimada"].diff()

    return sim


sim = simular_dias(prod_corte, sector, mes)

# -----------------------------
# KPIs EJECUTIVOS
# -----------------------------

pred_sel = sim.loc[sim["Dia_Empaque"] == dia_final, "Produccion_Estimada"].iloc[0]

brecha = prod_corte - pred_sel
rendimiento = (pred_sel / prod_corte) * 100

dia_optimo = int(sim.loc[sim["Produccion_Estimada"].idxmax(), "Dia_Empaque"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Producción estimada", f"{pred_sel:,.2f}")
c2.metric("Rendimiento del proceso", f"{rendimiento:,.2f}%")
c3.metric("Brecha vs corte", f"{brecha:,.2f}")
c4.metric("Día recomendado", dia_optimo)

st.divider()

# -----------------------------
# TABLA
# -----------------------------

st.subheader("Simulación por día de empaque (1 a 6)")

tabla = sim[[
    "Dia_Empaque",
    "Produccion_Estimada",
    "Brecha_vs_corte",
    "Rendimiento_%",
    "Delta_vs_dia1",
    "Cambio_marginal"
]]

st.dataframe(tabla, use_container_width=True)

st.info(
    f"Según el modelo, el mejor día es el **día {dia_optimo}**, "
    f"con una producción estimada de **{sim['Produccion_Estimada'].max():,.2f}**."
)

st.divider()

# -----------------------------
# PANEL VISUAL (4 GRÁFICAS)
# -----------------------------

st.subheader("Panel visual")

col1, col2 = st.columns(2)

# Gráfica 1 — Producción por día
fig1 = px.line(
    sim,
    x="Dia_Empaque",
    y="Produccion_Estimada",
    markers=True,
    title="Producción estimada por día"
)

col1.plotly_chart(fig1, use_container_width=True)

# Gráfica 2 — Brecha vs corte
fig2 = px.bar(
    sim,
    x="Dia_Empaque",
    y="Brecha_vs_corte",
    title="Brecha respecto al volumen de corte"
)

col2.plotly_chart(fig2, use_container_width=True)


# Segunda fila
col3, col4 = st.columns(2)

# Gráfica 3 — Rendimiento
fig3 = px.line(
    sim,
    x="Dia_Empaque",
    y="Rendimiento_%",
    markers=True,
    title="Rendimiento del proceso (%)"
)

col3.plotly_chart(fig3, use_container_width=True)

# Gráfica 4 — Cambio marginal
fig4 = px.bar(
    sim,
    x="Dia_Empaque",
    y="Cambio_marginal",
    title="Cambio marginal entre días"
)

col4.plotly_chart(fig4, use_container_width=True)
