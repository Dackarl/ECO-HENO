import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Config básica
# -----------------------------
st.set_page_config(page_title="EcoHeno 1.0", layout="wide")

st.title("EcoHeno 1.0 — Predicción de producción de heno")
st.caption("Cuadro maestro de decisión: predicción + simulación + visualización ejecutiva.")

# -----------------------------
# Cargar modelo y columnas
# -----------------------------
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_ecoheno.pkl")
    columnas = joblib.load("columnas_modelo.pkl")  # lista de columnas con las que entrenaste
    return modelo, columnas

modelo, columnas_modelo = cargar_modelo()

# -----------------------------
# (Opcional) Cargar df_modelo si existe
# -----------------------------
@st.cache_data
def cargar_df_modelo():
    try:
        dfm = pd.read_csv("df_modelo.csv")
        return dfm
    except Exception:
        return None

df_modelo = cargar_df_modelo()

# -----------------------------
# Sidebar (cuadro maestro)
# -----------------------------
st.sidebar.header("Cuadro maestro — Variables del ciclo")

prod_corte = st.sidebar.number_input(
    "Producción en corte",
    min_value=0.0,
    value=7000.0,
    step=50.0
)

dia_final = st.sidebar.slider(
    "Día final de empaque",
    min_value=1,
    max_value=6,
    value=5
)

sector = st.sidebar.selectbox(
    "Sector",
    options=[1,2,3,4,5,6,7,8],
    index=0
)

mes = st.sidebar.slider(
    "Mes",
    min_value=1,
    max_value=12,
    value=4
)

st.sidebar.divider()
btn = st.sidebar.button("Predecir")

# -----------------------------
# Funciones clave
# -----------------------------
def construir_X(prod_corte, dia_final, sector, mes, columnas):
    fila = {
        "PROD_CORTE": float(prod_corte),
        "DIA_FINAL": int(dia_final),
        "SECTOR": int(sector),
        "MES": int(mes),
    }
    X = pd.DataFrame([fila])

    # Asegurar exactamente las columnas del entrenamiento
    for c in columnas:
        if c not in X.columns:
            X[c] = 0

    X = X[columnas]
    return X

def predecir(modelo, X):
    pred = modelo.predict(X)
    # convertir a escalar estable
    pred = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
    return pred

def simular_dias(modelo, prod_corte, sector, mes, columnas, dias=range(1,7)):
    rows = []
    for d in dias:
        X = construir_X(prod_corte, d, sector, mes, columnas)
        p = predecir(modelo, X)
        rows.append({"Dia_Empaque": d, "Produccion_Predicha": p})
    sim = pd.DataFrame(rows)
    sim["Perdida_vs_dia1"] = sim["Produccion_Predicha"].iloc[0] - sim["Produccion_Predicha"]
    sim["Perdida_%"] = (sim["Perdida_vs_dia1"] / sim["Produccion_Predicha"].iloc[0]) * 100
    sim["Marginal_perdida"] = sim["Produccion_Predicha"].diff() * -1
    return sim

def plot_line_sim(sim):
    fig = plt.figure()
    plt.plot(sim["Dia_Empaque"], sim["Produccion_Predicha"], marker="o")
    plt.title("Impacto del día de empaque en la producción final")
    plt.xlabel("Día de empaque")
    plt.ylabel("Producción predicha")
    plt.grid(True, alpha=0.3)
    return fig

def plot_importancias(modelo, columnas):

    if not hasattr(modelo, "feature_importances_"):
        return None   # evita que Streamlit explote

    imp = pd.Series(
        modelo.feature_importances_,
        index=columnas
    ).sort_values(ascending=False).head(10)

    fig = plt.figure()
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title("Top variables más influyentes (importancia del modelo)")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.grid(True, axis="x", alpha=0.3)

    return fig

def plot_bar_conteo_sector(df_modelo):
    if df_modelo is None or "SECTOR" not in df_modelo.columns:
        return None

    conteo = df_modelo["SECTOR"].value_counts().sort_index()
    fig = plt.figure()
    plt.bar(conteo.index.astype(str), conteo.values)
    plt.title("Cantidad de ciclos por sector (dataset)")
    plt.xlabel("Sector")
    plt.ylabel("Número de ciclos")
    plt.grid(True, axis="y", alpha=0.3)
    return fig

def plot_scatter_real_vs_pred(df_modelo, modelo, columnas):
    if df_modelo is None:
        return None
    req = {"PROD_CORTE", "DIA_FINAL", "SECTOR", "MES", "PROD_FINAL"}
    if not req.issubset(set(df_modelo.columns)):
        return None

    X = df_modelo[["PROD_CORTE", "DIA_FINAL", "SECTOR", "MES"]].copy()
    for c in columnas:
        if c not in X.columns:
            X[c] = 0
    X = X[columnas]

    y_real = df_modelo["PROD_FINAL"].astype(float).values
    y_pred = modelo.predict(X)
    y_pred = np.array(y_pred).astype(float)

    fig = plt.figure()
    plt.scatter(y_real, y_pred, alpha=0.6)
    plt.title("Producción final real vs predicha (validación visual)")
    plt.xlabel("Real (PROD_FINAL)")
    plt.ylabel("Predicha")
    plt.grid(True, alpha=0.3)
    return fig

# -----------------------------
# Predicción principal
# -----------------------------
if btn:
    X_in = construir_X(prod_corte, dia_final, sector, mes, columnas_modelo)
    pred_final = predecir(modelo, X_in)

    perdida = float(prod_corte) - pred_final
    if prod_corte and prod_corte > 0:
        perdida_pct = (perdida / float(prod_corte)) * 100
    else:
        perdida_pct = 0.0

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Producción final estimada", f"{pred_final:,.2f}")
    c2.metric("Pérdida estimada (corte - final)", f"{perdida:,.2f}")
    c3.metric("Pérdida porcentual", f"{perdida_pct:,.2f}%")
    c4.metric("Día final de empaque", f"{dia_final}")

    st.divider()

    # Simulación por días
    sim = simular_dias(modelo, prod_corte, sector, mes, columnas_modelo)
    st.subheader("Simulación por día de empaque (1 a 6)")
    st.dataframe(sim, use_container_width=True)

    # -----------------------------
    # 4 GRÁFICAS
    # -----------------------------
    st.subheader("Panel visual (4 gráficas)")

    g1, g2 = st.columns(2)
    with g1:
        fig1 = plot_line_sim(sim)
        st.pyplot(fig1, use_container_width=True)

    with g2:
        fig2 = plot_importancias(modelo, columnas_modelo)
        if fig2 is None:
            st.info("Este modelo no expone importancias (feature_importances_).")
        else:
            st.pyplot(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig3 = plot_bar_conteo_sector(df_modelo)
        if fig3 is None:
            st.info("Si subes un archivo df_modelo.csv al repo, aquí mostramos conteos por sector.")
        else:
            st.pyplot(fig3, use_container_width=True)

    with g4:
        fig4 = plot_scatter_real_vs_pred(df_modelo, modelo, columnas_modelo)
        if fig4 is None:
            st.info("Si df_modelo.csv tiene PROD_FINAL y variables base, aquí se verá real vs predicho.")
        else:
            st.pyplot(fig4, use_container_width=True)

else:
    st.info("Ajusta los selectores en la izquierda y presiona Predecir.")
