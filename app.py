import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Para gráficas (requiere matplotlib en requirements.txt)
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
    columnas = joblib.load("columnas_modelo.pkl")  # lista EXACTA de columnas del entrenamiento
    return modelo, columnas

modelo, columnas_modelo = cargar_modelo()

# -----------------------------
# (Opcional) Cargar dataset de ciclos si existe
# (si NO lo tienes, no pasa nada)
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
# Sidebar — Cuadro maestro
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
    options=[1, 2, 3, 4, 5, 6, 7, 8],
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
# Funciones
# -----------------------------
def construir_X(prod_corte, dia_final, sector, mes, columnas):
    fila = {
        "PROD_CORTE": float(prod_corte),
        "DIA_FINAL": int(dia_final),
        "SECTOR": int(sector),
        "MES": int(mes),
    }
    X = pd.DataFrame([fila])

    # asegurar columnas exactas del entrenamiento
    for c in columnas:
        if c not in X.columns:
            X[c] = 0
    X = X[columnas]
    return X

def predecir_escalar(modelo, X):
    pred = modelo.predict(X)
    return float(np.asarray(pred).ravel()[0])

def simular_dias(modelo, prod_corte, sector, mes, columnas, dias=range(1, 7)):
    rows = []
    for d in dias:
        X = construir_X(prod_corte, d, sector, mes, columnas)
        prod_recuperable = predecir_escalar(modelo, X)
        rows.append({"Dia_Empaque": d, "Produccion_Recuperable": prod_recuperable})

    sim = pd.DataFrame(rows)

    # Métricas ejecutivas correctas
    sim["No_Recuperado"] = float(prod_corte) - sim["Produccion_Recuperable"]
    sim["Eficiencia_%"] = np.where(
        float(prod_corte) > 0,
        (sim["Produccion_Recuperable"] / float(prod_corte)) * 100,
        0.0
    )

    # Comparación vs día 1 (para ver el costo de empacar más tarde o más temprano)
    base_dia1 = sim["Produccion_Recuperable"].iloc[0]
    sim["Cambio_vs_dia1"] = sim["Produccion_Recuperable"] - base_dia1
    sim["Cambio_vs_dia1_%"] = np.where(
        base_dia1 > 0,
        (sim["Cambio_vs_dia1"] / base_dia1) * 100,
        0.0
    )

    # Pérdida marginal (cuánto cambia al pasar al siguiente día)
    sim["Cambio_marginal"] = sim["Produccion_Recuperable"].diff()

    return sim

def plot_line_sim(sim):
    fig = plt.figure()
    plt.plot(sim["Dia_Empaque"], sim["Produccion_Recuperable"], marker="o")
    plt.title("Impacto del día de empaque en la producción recuperable")
    plt.xlabel("Día de empaque")
    plt.ylabel("Producción recuperable (predicha)")
    plt.grid(True, alpha=0.3)
    return fig

def plot_importancias(modelo, columnas):
    if not hasattr(modelo, "feature_importances_"):
        return None

    imp = pd.Series(modelo.feature_importances_, index=columnas).sort_values(ascending=False).head(10)

    fig = plt.figure()
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title("Top 10 variables más influyentes (importancia del modelo)")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.grid(True, axis="x", alpha=0.3)
    return fig

def plot_conteo_sector(df_modelo):
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

def plot_real_vs_pred(df_modelo, modelo, columnas):
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
    y_pred = np.asarray(modelo.predict(X)).astype(float)

    fig = plt.figure()
    plt.scatter(y_real, y_pred, alpha=0.6)
    plt.title("Producción final real vs predicha (validación visual)")
    plt.xlabel("Real (PROD_FINAL)")
    plt.ylabel("Predicha")
    plt.grid(True, alpha=0.3)
    return fig

# -----------------------------
# Ejecución
# -----------------------------
if btn:
    X_in = construir_X(prod_corte, dia_final, sector, mes, columnas_modelo)
    prod_recuperable = predecir_escalar(modelo, X_in)

    no_recuperado = float(prod_corte) - prod_recuperable
    eficiencia = (prod_recuperable / float(prod_corte) * 100) if float(prod_corte) > 0 else 0.0

    # Recomendación automática: día con mayor producción recuperable
    sim = simular_dias(modelo, prod_corte, sector, mes, columnas_modelo)
    dia_recomendado = int(sim.loc[sim["Produccion_Recuperable"].idxmax(), "Dia_Empaque"])
    prod_max = float(sim["Produccion_Recuperable"].max())

    # KPIs ejecutivos
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Producción recuperable (estimada)", f"{prod_recuperable:,.2f}")
    c2.metric("Eficiencia del corte", f"{eficiencia:,.2f}%")
    c3.metric("Volumen no recuperado", f"{no_recuperado:,.2f}")
    c4.metric("Día recomendado", f"{dia_recomendado}")

    st.divider()

    # Tabla de simulación
    st.subheader("Simulación por día de empaque (1 a 6)")
    st.dataframe(sim, use_container_width=True)

    # Mensaje ejecutivo rápido
    st.info(f"Según el modelo, el mejor día (máxima producción recuperable) es el **día {dia_recomendado}** "
            f"con una producción recuperable aproximada de **{prod_max:,.2f}**.")

    # -----------------------------
    # Panel visual (4 gráficas)
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
        fig3 = plot_conteo_sector(df_modelo)
        if fig3 is None:
            st.info("Si subes un archivo `df_modelo.csv` al repo, aquí mostramos el conteo por sector.")
        else:
            st.pyplot(fig3, use_container_width=True)

    with g4:
        fig4 = plot_real_vs_pred(df_modelo, modelo, columnas_modelo)
        if fig4 is None:
            st.info("Si `df_modelo.csv` tiene `PROD_FINAL` y variables base, aquí se verá Real vs Predicho.")
        else:
            st.pyplot(fig4, use_container_width=True)

else:
    st.info("Ajusta los selectores a la izquierda y presiona **Predecir**.")
