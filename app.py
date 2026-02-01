import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# 1) CONFIG
# ============================================================
st.set_page_config(page_title="EcoHeno 1.0", layout="wide")

st.title("EcoHeno 1.0 — Predicción de producción de heno")
st.caption("Cuadro maestro de decisión: predicción + simulación + visualización ejecutiva.")

# ============================================================
# 2) CARGA DE MODELO + COLUMNAS (para evitar el error de sklearn)
# ============================================================
@st.cache_resource
def cargar_artifacts():
    modelo = joblib.load("modelo_ecoheno.pkl")
    columnas = joblib.load("columnas_modelo.pkl")  # lista exacta del entrenamiento
    return modelo, columnas

modelo, columnas_modelo = cargar_artifacts()

with st.expander("Columnas que espera el modelo (debug)"):
    st.write(columnas_modelo)

# ============================================================
# 3) CONSTRUIR X CON COLUMNAS EXACTAS
# ============================================================
def construir_X(prod_corte, dia_final, sector, mes):
    """
    Crea el vector de entrada X con las MISMAS columnas del entrenamiento.
    Rellena con mapeo robusto para no romperse si los nombres cambian.
    """
    X = pd.DataFrame(np.zeros((1, len(columnas_modelo))), columns=columnas_modelo)

    # Mapeo robusto: intenta encajar con varios nombres posibles
    alias = {
        # producción en corte
        "PROD_CORTE": prod_corte,
        "Produccion_corte": prod_corte,
        "produccion_corte": prod_corte,

        # día final empaque
        "DIA_FINAL": dia_final,
        "Dia_final": dia_final,
        "DIA_EMPAQUE": dia_final,
        "Dia_Empaque": dia_final,
        "Dia_Empaque_Final": dia_final,

        # sector
        "SECTOR": sector,
        "Sector": sector,
        "sector": sector,

        # mes
        "MES": mes,
        "Mes": mes,
        "mes": mes,
    }

    # Solo llena las columnas que existan en columnas_modelo
    for col in X.columns:
        if col in alias:
            X.loc[0, col] = alias[col]

    return X

# ============================================================
# 4) PREDICCIÓN
# ============================================================
def predecir_produccion(prod_corte, dia_final, sector, mes):
    X = construir_X(prod_corte, dia_final, sector, mes)
    pred = float(modelo.predict(X)[0])
    # por si el modelo devuelve negativos por ruido
    return max(pred, 0.0)

# ============================================================
# 5) SIDEBAR (selectores)
# ============================================================
st.sidebar.header("Cuadro maestro — Variables del ciclo")

prod_corte = st.sidebar.number_input(
    "Producción en corte",
    min_value=0.0,
    value=8000.0,
    step=100.0
)

dia_final = st.sidebar.slider(
    "Día final de empaque",
    min_value=1,
    max_value=6,
    value=3
)

sector = st.sidebar.selectbox(
    "Sector",
    options=[1, 2, 3, 4, 5, 6],
    index=0
)

mes = st.sidebar.slider(
    "Mes",
    min_value=1,
    max_value=12,
    value=1
)

# botón (opcional, pero se ve profesional)
btn = st.sidebar.button("Predecir")

# ============================================================
# 6) SIMULACIÓN (días 1 a 6)
# ============================================================
def simular_dias(prod_corte, sector, mes):
    filas = []
    for d in range(1, 7):
        pred = predecir_produccion(prod_corte, d, sector, mes)
        filas.append({
            "Dia_Empaque": d,
            "Produccion_Estimada": pred
        })

    sim = pd.DataFrame(filas)

    # Métricas claras (sin “No_recuperado” confuso)
    sim["Perdida_%"] = (sim["Produccion_Estimada"] / prod_corte) * 100

    base = sim.loc[sim["Dia_Empaque"] == 1, "Produccion_Estimada"].iloc[0]
    sim["Delta_vs_dia1"] = sim["Produccion_Estimada"] - base
    sim["Cambio_marginal"] = sim["Produccion_Estimada"].diff()

    return sim

sim = simular_dias(prod_corte, sector, mes)

# ============================================================
# 7) KPIs (cuadro superior)
# ============================================================
pred_sel = sim.loc[sim["Dia_Empaque"] == dia_final, "Produccion_Estimada"].iloc[0]
brecha = prod_corte - pred_sel
rendimiento = (pred_sel / prod_corte) * 100

dia_optimo = int(sim.loc[sim["Produccion_Estimada"].idxmax(), "Dia_Empaque"])
mejor_pred = float(sim["Produccion_Estimada"].max())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Producción estimada", f"{pred_sel:,.2f}")
c2.metric("Rendimiento del proceso", f"{(100-rendimiento):,.2f}%")
c3.metric("Brecha vs corte", f"{brecha:,.2f}")
c4.metric("Día recomendado", f"{dia_optimo}")

st.divider()

# ============================================================
# 8) TABLA DE SIMULACIÓN
# ============================================================
st.subheader("Simulación por día de empaque (1 a 6)")
st.dataframe(sim, use_container_width=True)

st.info(
    f"Según el modelo, el mejor día es el **día {dia_optimo}**, "
    f"con una producción estimada de **{mejor_pred:,.2f}**."
)

st.divider()

# ============================================================
# 9) PANEL VISUAL (4 GRÁFICAS)
# ============================================================
st.subheader("Panel visual (4 gráficas)")

col1, col2 = st.columns(2)

fig1 = px.line(
    sim,
    x="Dia_Empaque",
    y="Produccion_Estimada",
    markers=True,
    title="Producción estimada por día"
)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(
    sim,
    x="Dia_Empaque",
    y="Produccion luego de perdidas",
    title="Brecha respecto al volumen de corte"
)
col2.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

fig3 = px.line(
    sim,
    x="Dia_Empaque",
    y="Perdida_%",
    markers=True,
    title="Rendimiento del proceso (%)"
)
col3.plotly_chart(fig3, use_container_width=True)

fig4 = px.bar(
    sim,
    x="Dia_Empaque",
    y="Cambio_marginal",
    title="Cambio marginal entre días"
)
col4.plotly_chart(fig4, use_container_width=True)

