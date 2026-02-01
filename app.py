import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Configuración inicial de la página
st.set_page_config(page_title="Tablero Maestro ECO HENO", layout="wide")

# 1. CARGA DE RECURSOS
# Se cargan los archivos generados en el entrenamiento (RandomForest y lista de columnas)
@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load("modelo_ecoheno.pkl")
        columnas = joblib.load("columnas_modelo.pkl")
        return modelo, columnas
    except Exception as e:
        st.error("Error: Asegurese de tener 'modelo_ecoheno.pkl' y 'columnas_modelo.pkl' en el repositorio.")
        return None, None

modelo, columnas = cargar_recursos()

# 2. FUNCIONES DE CÁLCULO
# predecir_produccion: Calcula el valor puntual basado en las 4 variables maestras
def predecir_produccion(modelo, columnas, prod_corte, dia_final, sector, mes):
    df_input = pd.DataFrame([{
        "PROD_CORTE": prod_corte,
        "DIA_FINAL": dia_final,
        "SECTOR": sector,
        "MES": mes
    }])
    df_input = df_input.reindex(columns=columnas, fill_value=0)
    return float(modelo.predict(df_input)[0])

# simular_ciclo: Genera la tabla comparativa del dia 1 al 6 para el Cuadro Maestro
def simular_ciclo(modelo, columnas, prod_corte, sector, mes):
    datos = []
    for d in range(1, 7):
        p = predecir_produccion(modelo, columnas, prod_corte, d, sector, mes)
        datos.append({"Dia_Empaque": d, "Produccion_Predicha": round(p, 2)})
    
    df = pd.DataFrame(datos)
    # Cálculo de indicadores de eficiencia y pérdida
    df["Perdida_Acumulada"] = df["Produccion_Predicha"].iloc[0] - df["Produccion_Predicha"]
    df["Eficiencia_%"] = (df["Produccion_Predicha"] / prod_corte) * 100
    df["Cambio_Marginal"] = df["Produccion_Predicha"].diff()
    return df

# 3. INTERFAZ DE USUARIO (CUADRO MAESTRO)
st.title("Control de Produccion ECO HENO")

if modelo:
    # Sidebar para ingreso de datos de producción
    st.sidebar.header("Entrada de Datos")
    prod_corte = st.sidebar.number_input("Produccion Inicial (Corte)", min_value=0.0, value=7500.0)
    sector = st.sidebar.selectbox("Sector", options=list(range(1, 9)), index=2)
    mes = st.sidebar.selectbox("Mes de Operacion", options=list(range(1, 13)), index=5)
    dia_objetivo = st.sidebar.slider("Dia de Empaque Objetivo", 1, 6, 4)

    # Procesamiento
    df_maestro = simular_ciclo(modelo, columnas, prod_corte, sector, mes)
    pred_actual = df_maestro.loc[df_maestro["Dia_Empaque"] == dia_objetivo, "Produccion_Predicha"].values[0]
    eficiencia_actual = df_maestro.loc[df_maestro["Dia_Empaque"] == dia_objetivo, "Eficiencia_%"].values[0]

    # 4. VISUALIZACIÓN DE INDICADORES
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediccion Final", f"{pred_actual:.2f} kg")
    with col2:
        st.metric("Eficiencia Estimada", f"{eficiencia_actual:.1f}%")
    with col3:
        st.metric("Sector Seleccionado", f"Sector {sector}")

    # Gráfico de tendencia
    st.subheader("Tendencia de Produccion por Dia")
    fig = px.line(df_maestro, x="Dia_Empaque", y="Produccion_Predicha", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Tabla Maestra
    st.subheader("Cuadro Maestro de Indicadores")
    st.dataframe(df_maestro, use_container_width=True)

    # Nota Técnica: El umbral de elasticidad ayuda a decidir el momento óptimo de empaque
    # Basado en el mayor cambio marginal negativo
    id_optimo = df_maestro["Cambio_Marginal"].abs().idxmax() if not df_maestro["Cambio_Marginal"].isnull().all() else 0
    dia_optimo = df_maestro.loc[id_optimo, "Dia_Empaque"]
    
    st.info(f"Nota: El mayor cambio en la tasa de secado se observa en el Dia {dia_optimo}.")