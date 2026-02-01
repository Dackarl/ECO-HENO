import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Dashboard ECO HENO", layout="wide")

@st.cache_data
def load_and_process_data():
    """Procesamiento completo automatico desde Excel crudo"""
    
    # Verificar archivo existe
    if not os.path.exists('BASE_ECO_HENO_F.xlsx'):
        st.error("ERROR: BASE_ECO_HENO_F.xlsx no encontrado en el folder")
        st.stop()
    
    # 1. Cargar Excel crudo
    df = pd.read_excel('BASE_ECO_HENO_F.xlsx')
    
    # 2. FASE 1: Unificacion largo (codigo exacto del notebook)
    base_cols = ["AÑO", "MES", "DÍAS", "FECHA COMPLETA"]
    suffixes = ["", ".1", ".2", ".3"]
    
    bloques = []
    for suf in suffixes:
        act = f"ACTIVIDAD DEL DÍA SECTOR{suf}"
        sec = f"SECTOR{suf}"
        
        if sec in df.columns and act in df.columns:
            cols = base_cols + [act, sec]
            tmp = df[cols].copy()
            
            notas = f"NOTAS{suf}"
            prod = f"PRODUCCION DE HENO SECTOR{suf}"
            tmp["NOTAS"] = df[notas] if notas in df.columns else np.nan
            tmp["PRODUCCION_HENO"] = df[prod] if prod in df.columns else np.nan
            
            tmp = tmp.rename(columns={act: "ACTIVIDAD", sec: "SECTOR"})
            bloques.append(tmp)
    
    df_unificada = pd.concat(bloques, ignore_index=True)
    
    # Limpieza tipos
    df_unificada["FECHA COMPLETA"] = pd.to_datetime(df_unificada["FECHA COMPLETA"], errors='coerce')
    df_unificada["SECTOR"] = pd.to_numeric(df_unificada["SECTOR"], errors='coerce')
    df_unificada["PRODUCCION_HENO"] = pd.to_numeric(df_unificada["PRODUCCION_HENO"], errors='coerce')
    df_unificada = df_unificada.dropna(subset=["SECTOR", "FECHA COMPLETA"])
    df_unificada["SECTOR"] = df_unificada["SECTOR"].astype(int)
    
    # 3. FASE 2: Normalizacion y ETAPA
    df_unificada["ACTIVIDAD_NORM"] = (
        df_unificada["ACTIVIDAD"].astype(str).str.upper()
        .str.replace("–", "-").str.replace("—", "-")
        .str.replace("Á", "A").str.replace("É", "E")
        .str.replace("Í", "I").str.replace("Ó", "O").str.replace("Ú", "U")
        .str.strip()
    )
    
    df_unificada["ETAPA"] = np.select([
        df_unificada["ACTIVIDAD_NORM"].str.contains("CORTE"),
        df_unificada["ACTIVIDAD_NORM"].str.contains("SECAD"),
        df_unificada["ACTIVIDAD_NORM"].str.contains("VOLT"),
        df_unificada["ACTIVIDAD_NORM"].str.contains("EMPAQ")
    ], ["CORTE", "SECADO", "VOLTEO", "EMPAQUE"], "OTRA")
    
    # DIA_EMPAQUE
    df_unificada["DIA_EMPAQUE"] = np.nan
    mask_empaque = df_unificada["ETAPA"] == "EMPAQUE"
    if mask_empaque.sum() > 0:
        df_unificada.loc[mask_empaque, "DIA_EMPAQUE"] = (
            df_unificada.loc[mask_empaque, "ACTIVIDAD_NORM"]
            .str.extract(r"DIA\s*(\d+)").astype(float)
        )
    
    return df_unificada

# ==================== INICIO DASHBOARD ====================
st.title("Dashboard ECO HENO - Q1")

# Cargar datos
df = load_and_process_data()
st.success(f"Datos procesados: {len(df)} filas | {df['SECTOR'].nunique()} sectores")

# Cargar modelo (opcional)
try:
    model = joblib.load('rf_modelo.pkl')
    st.info("Modelo Random Forest cargado")
except:
    model = None
    st.warning("Modelo no encontrado - usando eficiencia media 9.6%")

# ==================== KPIs ====================
empaque_data = df[df["ETAPA"] == "EMPAQUE"].copy()
corte_data = df[df["ETAPA"] == "CORTE"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Registros", len(df))
col2.metric("Registros Empaque", len(empaque_data))
col3.metric("Sectores", df['SECTOR'].nunique())
col4.metric("Prod Media Corte", f"{corte_data['PRODUCCION_HENO'].mean():.0f} kg")

# ==================== GRÁFICOS ====================
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="ETAPA", color="SECTOR", 
                       title="Distribucion por Etapas")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if len(empaque_data) > 0:
        fig2 = px.box(empaque_data, x="DIA_EMPAQUE", y="PRODUCCION_HENO",
                     title="Produccion por Dia Empaque")
        st.plotly_chart(fig2, use_container_width=True)

# ==================== TABLA DATOS PROCESADOS ====================
st.subheader("Datos Procesados")
st.dataframe(df[['SECTOR', 'ETAPA', 'DIA_EMPAQUE', 'PRODUCCION_HENO']].head(20))

# ==================== SIMULADOR ====================
st.subheader("Simulador Produccion")
prod_corte = st.slider("Produccion Corte kg", 5000, 10000, 7500)

if model:
    pred_final = model.predict([[prod_corte]])[0]
else:
    pred_final = 0.096 * prod_corte

st.metric("Produccion Final Predicha", f"{pred_final:.0f} kg")
st.metric("Eficiencia", f"{pred_final/prod_corte*100:.1f}%")

