# app.py - Dashboard ECO HENO TOTALMENTE CONECTADO AL NOTEBOOK
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(page_title="🌾 Dashboard ECO HENO", layout="wide")

# ==================== CARGA AUTOMÁTICA DESDE NOTEBOOK ====================
@st.cache_data
def load_real_data():
    """CARGA DIRECTA de los archivos del notebook original"""
    
    # 1. Cargar base cruda (igual que notebook)
    df = pd.read_excel('BASE_ECO_HENO_F.xlsx')
    
    # 2. EJECUTAR FASE 1 AUTOMÁTICA (unificación largo)
    base_cols = ["AÑO", "MES", "DÍAS", "FECHA COMPLETA"]
    suffixes = ["", ".1", ".2", ".3"]
    
    bloques = []
    for suf in suffixes:
        act = f"ACTIVIDAD DEL DÍA SECTOR{suf}"
        sec = f"SECTOR{suf}"
        notas = f"NOTAS{suf}"
        prod = f"PRODUCCION DE HENO SECTOR{suf}"
        
        if sec in df.columns and act in df.columns:
            cols = base_cols + [act, sec]
            tmp = df[cols].copy()
            tmp["NOTAS"] = df[notas] if notas in df.columns else np.nan
            tmp["PRODUCCION_HENO"] = df[prod] if prod in df.columns else np.nan
            tmp = tmp.rename(columns={act: "ACTIVIDAD", sec: "SECTOR"})
            bloques.append(tmp)
    
    df_unificada = pd.concat(bloques, ignore_index=True)
    df_unificada["FECHA COMPLETA"] = pd.to_datetime(df_unificada["FECHA COMPLETA"], errors='coerce')
    df_unificada["SECTOR"] = pd.to_numeric(df_unificada["SECTOR"], errors='coerce')
    df_unificada["PRODUCCION_HENO"] = pd.to_numeric(df_unificada["PRODUCCION_HENO"], errors='coerce')
    df_unificada = df_unificada.dropna(subset=["SECTOR", "FECHA COMPLETA"]).copy()
    df_unificada["SECTOR"] = df_unificada["SECTOR"].astype(int)
    
    # 3. FASE 2 AUTOMÁTICA (ACTIVIDAD_NORM + ETAPA + DIA_EMPAQUE)
    df_unificada["ACTIVIDAD_NORM"] = (
        df_unificada["ACTIVIDAD"].astype(str)
        .str.upper().str.replace("–", "-").str.replace("—", "-")
        .str.replace("Á", "A").str.replace("É", "E").str.replace("Í", "I")
        .str.replace("Ó", "O").str.replace("Ú", "U").str.strip()
    )
    
    df_unificada["ETAPA"] = np.select([
        df_unificada["ACTIVIDAD_NORM"].str.contains("CORTE", na=False),
        df_unificada["ACTIVIDAD_NORM"].str.contains("SECAD", na=False),
        df_unificada["ACTIVIDAD_NORM"].str.contains("VOLT", na=False),
        df_unificada["ACTIVIDAD_NORM"].str.contains("EMPAQ", na=False)
    ], ["CORTE", "SECADO", "VOLTEO", "EMPAQUE"], default="OTRA")
    
    # DIA_EMPAQUE automático
    df_unificada["DIA_EMPAQUE"] = np.nan
    mask_empaque = df_unificada["ETAPA"] == "EMPAQUE"
    df_unificada.loc[mask_empaque, "DIA_EMPAQUE"] = (
        df_unificada.loc[mask_empaque, "ACTIVIDAD_NORM"]
        .str.extract(r"DIA\s*(\d+)").astype(float)
    )
    
    # 4. CARGAR MODELO (debes guardar en notebook: joblib.dump(model, 'rf_modelo.pkl'))
    try:
        model = joblib.load('rf_modelo.pkl')
    except:
        # Modelo dummy si no existe
        model = lambda x: 0.096 * x
        st.warning("⚠️ Modelo dummy activo - GUARDA rf_modelo.pkl desde notebook")
    
    return df_unificada, model

# ==================== DASHBOARD PRINCIPAL ====================
st.title("🌾 **Dashboard ECO HENO - Automático**")

df_real, model = load_real_data()
st.success(f"✅ **Datos cargados**: {len(df_real):,} filas | {df_real['SECTOR'].nunique()} sectores")

# KPIs REALES
empaque_data = df_real[df_real["ETAPA"] == "EMPAQUE"].copy()
if len(empaque_data) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros Empaque", len(empaque_data))
    col2.metric("Producción Media Día 1", 
               f"{empaque_data[empaque_data['DIA_EMPAQUE']==1]['PRODUCCION_HENO'].mean():.0f} kg")
    col3.metric("Días Máx Observados", f"{empaque_data['DIA_EMPAQUE'].max():.0f}")

# ==================== VISUALIZACIÓN AUTOMÁTICA ====================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Por Día Empaque", "⚙️ Simulador"])

with tab1:
    # Distribución etapas
    fig_etapas = px.histogram(df_real, x="ETAPA", color="SECTOR",
                             title="Distribución por Etapas y Sector")
    st.plotly_chart(fig_etapas, use_container_width=True)
    
    # Producción por etapa
    fig_prod = px.box(df_real, x="ETAPA", y="PRODUCCION_HENO", color="SECTOR",
                     title="Producción por Etapa (Boxplot)")
    st.plotly_chart(fig_prod, use_container_width=True)

with tab2:
    if len(empaque_data) > 0:
        # Gráfico PERDIDA por día (usando modelo)
        empaque_data['PROD_PRED_DIA1'] = model(empaque_data['PRODUCCION_HENO'])
        empaque_data['PERDIDA_KG'] = empaque_data['PROD_PRED_DIA1'] - empaque_data['PRODUCCION_HENO']
        
        fig_perdidas = px.scatter(empaque_data, x="DIA_EMPAQUE", y="PERDIDA_KG",
                                 color="SECTOR", size="PRODUCCION_HENO",
                                 title="🔥 Pérdidas Reales por Día de Empaque")
        st.plotly_chart(fig_perdidas, use_container_width=True)
        
        st.dataframe(empaque_data[['SECTOR', 'DIA_EMPAQUE', 'PRODUCCION_HENO', 'PERDIDA_KG']].round(1))

with tab3:
    # SIMULADOR REAL
    st.subheader("🎛️ Simulador Producción Corte → Final")
    prod_corte_input = st.slider("🌾 Producción en Corte (kg)", 5000, 10000, 7500)
    prod_final = model(prod_corte_input)
    
    col1, col2 = st.columns(2)
    col1.metric("🎁 Producción Final Predicha", f"{prod_final:.0f} kg")
    col2.metric("📉 Eficiencia", f"{prod_final/prod_corte_input*100:.1f}%")
    
    st.info(f"**Modelo activo**: {type(model).__name__} | Datos: {len(df_real)} registros")

st.markdown("---")
st.caption("🛡️ Dashboard 100% conectado al notebook | Actualización automática")

