import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================
# UI labels (EN)
# =========================
UI_COLS = {
    "Dia_Empaque": "Packing day",
    "Produccion_Estimada": "Estimated output (kg)",
    "Recovered_%": "Recovered (%)",
    "Loss_%": "Process loss (%)",
    "Cambio_marginal": "Marginal change (kg)",
    "Perdida_real": "Absolute loss (kg)",
    "Produccion_Acumulada": "Cumulative output (kg)",
    "Produccion_Corte": "Cut volume (kg)"
}

# ============================================================
# 1) CONFIG
# ============================================================
st.set_page_config(page_title="EcoHeno 2.0", layout="wide")

st.title("🌾 EcoHeno 2.0 — Hay Production Forecasting")
st.caption("Decision dashboard: prediction + simulation + executive visualization.")

# ============================================================
# 2) LOAD MODEL + COLUMNS (avoid sklearn feature mismatch)
# ============================================================
@st.cache_resource
def cargar_artifacts():
    modelo = joblib.load("modelo_ecoheno.pkl")
    columnas = joblib.load("columnas_modelo.pkl")  # exact list used in training
    return modelo, columnas

modelo, columnas_modelo = cargar_artifacts()

with st.expander("Model expected columns (debug)"):
    st.write(columnas_modelo)

# ============================================================
# 3) BUILD X WITH EXACT TRAINING COLUMNS
# ============================================================
def construir_X(prod_corte, dia_final, sector, mes):
    X = pd.DataFrame(np.zeros((1, len(columnas_modelo))), columns=columnas_modelo)

    alias = {
        # cut volume
        "PROD_CORTE": prod_corte,
        "Produccion_corte": prod_corte,
        "produccion_corte": prod_corte,

        # packing day
        "DIA_FINAL": dia_final,
        "Dia_final": dia_final,
        "DIA_EMPAQUE": dia_final,
        "Dia_Empaque": dia_final,
        "Dia_Empaque_Final": dia_final,

        # sector
        "SECTOR": sector,
        "Sector": sector,
        "sector": sector,

        # month
        "MES": mes,
        "Mes": mes,
        "mes": mes,
    }

    for col in X.columns:
        if col in alias:
            X.loc[0, col] = alias[col]

    return X

# ============================================================
# 4) PREDICTION
# ============================================================
def predecir_produccion(prod_corte, dia_final, sector, mes):
    X = construir_X(prod_corte, dia_final, sector, mes)
    pred = float(modelo.predict(X)[0])
    return max(pred, 0.0)

# ============================================================
# 5) SIDEBAR (selectors)
# ============================================================
st.sidebar.header("Master controls — Cycle variables")

CAPACIDAD_MAXIMA = 9000.0

prod_corte = st.sidebar.number_input(
    "Cut volume (kg)",
    min_value=0.0,
    max_value=CAPACIDAD_MAXIMA,
    value=min(6000.0, CAPACIDAD_MAXIMA),
    step=100.0
)

dia_final = st.sidebar.slider(
    "Final packing day",
    min_value=1,
    max_value=6,
    value=3
)

sector = st.sidebar.selectbox(
    "Sector",
    options=[1, 2, 3, 4, 5, 6, 7, 8],
    index=0
)

mes = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=1
)

st.sidebar.button("Run prediction")  # purely UI (Streamlit reruns anyway)

# ============================================================
# 6) SIMULATION (days 1 to 6)
# ============================================================
def simular_dias(prod_corte, sector, mes):
    filas = []
    for d in range(1, 7):
        pred = predecir_produccion(prod_corte, d, sector, mes)
        filas.append({"Dia_Empaque": d, "Produccion_Estimada": pred})

    sim = pd.DataFrame(filas)

    # Recovered (%) and Loss (%) — this fixes the “10% looks awful” issue
    sim["Recovered_%"] = (sim["Produccion_Estimada"] / prod_corte) * 100 if prod_corte > 0 else 0.0
    sim["Loss_%"] = 100 - sim["Recovered_%"]

    # Marginal change (kg)
    sim["Cambio_marginal"] = sim["Produccion_Estimada"].diff()

    return sim

sim = simular_dias(prod_corte, sector, mes)

# ============================================================
# 7) KPIs (top panel)
# ============================================================
pred_sel = float(sim.loc[sim["Dia_Empaque"] == dia_final, "Produccion_Estimada"].iloc[0])
gap_vs_cut = float(prod_corte - pred_sel)

recovered_pct = float((pred_sel / prod_corte) * 100) if prod_corte > 0 else 0.0
loss_pct = float(100 - recovered_pct)

dia_optimo = int(sim.loc[sim["Produccion_Estimada"].idxmax(), "Dia_Empaque"])
best_pred = float(sim["Produccion_Estimada"].max())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimated output (kg)", f"{pred_sel:,.2f}")
c2.metric("Cut-day loss (%)", f"{loss_pct:,.2f}%")   # ✅ NOW it’s loss, not recovered
c3.metric("Gap vs cut (kg)", f"{gap_vs_cut:,.2f}")
c4.metric("Recommended day", f"{dia_optimo}")

st.divider()

# ============================================================
# 8) SIMULATION TABLE
# ============================================================
st.subheader("Packing-day simulation (1 to 6)")

table_cols = ["Dia_Empaque", "Produccion_Estimada", "Recovered_%", "Loss_%", "Cambio_marginal"]
st.dataframe(sim[table_cols].rename(columns=UI_COLS), use_container_width=True)

st.info(
    f"Based on the model, the best option is **day {dia_optimo}**, "
    f"with an estimated output of **{best_pred:,.2f} kg**."
)

st.divider()

# ============================================================
# 9) VISUAL PANEL
# ============================================================
st.subheader("Operational visual panel")

# Real loss in kg
sim["Perdida_real"] = prod_corte - sim["Produccion_Estimada"]

col1, col2 = st.columns(2)

# 1 — Loss percentage (LEFT)
fig_loss = px.line(
    sim,
    x="Dia_Empaque",
    y="Loss_%",
    markers=True,
    title="Process loss percentage",
    labels={"Dia_Empaque": "Packing day", "Loss_%": "Process loss (%)"}
)
col1.plotly_chart(fig_loss, use_container_width=True)

# 2 — Impact (RIGHT)
fig_impact = px.scatter(
    sim,
    x="Dia_Empaque",
    y="Produccion_Estimada",
    size="Produccion_Estimada",
    title="Impact of packing day on estimated output",
    labels={"Dia_Empaque": "Packing day", "Produccion_Estimada": "Estimated output (kg)"}
)
fig_impact.update_traces(
    hovertemplate="Packing day=%{x}<br>Estimated output=%{y:.2f} kg<extra></extra>"
)
col2.plotly_chart(fig_impact, use_container_width=True)

# 3 — Cumulative progress (bars)
st.warning(
    "Operational note: cumulative output is limited by the packing capacity of a single operator. "
    "Higher cut volumes would require expanding installed capacity."
)

sim["Produccion_Corte"] = prod_corte
sim["Produccion_Acumulada"] = sim["Produccion_Estimada"].cumsum()
sim["Produccion_Acumulada"] = sim["Produccion_Acumulada"].clip(upper=prod_corte)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=["Cut volume"],
    y=[prod_corte],
    name="Cut volume"
))

fig.add_trace(go.Bar(
    x=[f"Day {d}" for d in sim["Dia_Empaque"]],
    y=sim["Produccion_Acumulada"],
    name="Cumulative output"
))

fig.update_layout(
    title="Cumulative packing progress",
    barmode="relative",
    xaxis_title="Process stage",
    yaxis_title="Output (kg)",
    legend_title="Series"
)

st.plotly_chart(fig, use_container_width=True)

st.warning(
    "Operational note: the predictive analysis indicates a packing-capacity constraint under higher cut-volume scenarios. "
    "This suggests the current process relies on limited operating capacity, which may create congestion if production increases. "
    "Therefore, the system not only estimates output but also helps anticipate operational adjustments to sustain continuity."
)

