import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests

# ==========================================================
# Configuração
# ==========================================================
API_URL = "https://kunzcmsbrljxdeyeeqgi.supabase.co/functions/v1/get-timepoints"

st.set_page_config(page_title="N-of-1 Oncology Dashboard", layout="wide")

st.title("🧬 N-of-1 Oncology Dashboard – Tumor Burden Predictor")
st.markdown("""
Este dashboard lê automaticamente os dados do paciente direto do banco de dados,
sem necessidade de upload de CSV.
""")

# ==========================================================
# 1. Seleção do Caso + Carregamento
# ==========================================================
st.header("📂 1. Selecione o Caso")

case_id = st.text_input(
    "Digite o case_id do paciente:", 
    placeholder="ex: 91e64fea-3d77-4a53-a4cb-43b6fe87123a"
)

if not case_id:
    st.info("👆 Digite o ID do caso para começar")
    st.stop()

# Buscar dados via API REST
try:
    response = requests.get(f"{API_URL}?case_id={case_id}")
    response.raise_for_status()
    
    result = response.json()
    
    if "error" in result:
        st.error(f"Erro: {result['error']}")
        st.stop()
    
    if len(result["data"]) == 0:
        st.error("Nenhum timepoint encontrado para esse case_id.")
        st.stop()
    
    df = pd.DataFrame(result["data"])
    
    # Converter colunas
    df["date"] = pd.to_datetime(df["date"])
    df["sum_mm"] = pd.to_numeric(df["sum_mm"], errors="coerce")
    
    # Remover linhas sem tumor_burden
    df = df.dropna(subset=["sum_mm"])
    
    if len(df) == 0:
        st.warning("Não há dados de tumor burden disponíveis para este caso.")
        st.stop()
    
    st.write("### 🔍 Timepoints carregados:")
    st.dataframe(df[["date", "sum_mm", "event_date", "source_type"]])
    
except requests.exceptions.RequestException as e:
    st.error(f"Erro ao conectar com a API: {str(e)}")
    st.stop()

# ==========================================================
# 2. Modelagem preditiva
# ==========================================================
st.header("🧠 2. Modelagem Preditiva")

df = df.sort_values("date")
T0 = df["date"].min()
df["days"] = (df["date"] - T0).dt.days

X = df["days"].values.reshape(-1, 1)
y = df["sum_mm"].values

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

col1, col2 = st.columns(2)
with col1:
    st.metric("Slope (mm/dia)", f"{slope:.4f}")
with col2:
    st.metric("Intercept", f"{intercept:.2f}")

future_days = st.multiselect(
    "Selecione horizontes de previsão (em dias):",
    [30, 60, 90, 120, 180, 240, 360],
    default=[30, 90, 180]
)

if not future_days:
    st.warning("Selecione ao menos um horizonte de previsão.")
    st.stop()

preds = {
    "days": future_days,
    "predicted_sum_mm": [intercept + slope * d for d in future_days],
    "date": [(T0 + timedelta(days=d)).strftime("%Y-%m-%d") for d in future_days]
}

pred_df = pd.DataFrame(preds)

st.write("### 📈 Previsões:")
st.table(pred_df)

# ==========================================================
# 3. Gráfico
# ==========================================================
st.header("📊 3. Curva — Real vs Predito")

fig, ax = plt.subplots(figsize=(12, 6))

# Converter dates para plot
pred_dates = pd.to_datetime(pred_df["date"])

ax.plot(df["date"], df["sum_mm"], marker="o", linewidth=3, markersize=8, label="Real", color="#2563eb")
ax.plot(pred_dates, pred_df["predicted_sum_mm"], marker="x", linestyle="--", linewidth=2, markersize=10, label="Predito", color="#dc2626")

ax.set_xlabel("Data", fontsize=12)
ax.set_ylabel("Tumor burden (mm)", fontsize=12)
ax.set_title("N-of-1 – Evolução do Tumor", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# 4. Probabilidade heurística de progressão
# ==========================================================
st.header("📉 4. Probabilidade de Progressão")

prob_pd = float(np.clip((slope / 1.0) * 100 + 50, 0, 100))

st.metric(
    label="Probabilidade aproximada de progressão nos próximos 6 meses",
    value=f"{prob_pd:.1f}%",
    delta=f"Slope: {slope:.4f} mm/dia"
)

if slope > 0:
    st.error("⚠️ Tendência de crescimento do tumor detectada")
elif slope < -0.5:
    st.success("✅ Tendência de redução significativa do tumor")
else:
    st.info("ℹ️ Tumor burden estável")

st.caption("""
*Obs.: baseado no slope atual; para probabilidade real use modelo Bayesiano.*
""")

# ==========================================================
# 5. Exportação
# ==========================================================
st.header("📤 5. Exportar Previsões")

csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar CSV de Previsões",
    data=csv,
    file_name=f"n_of_1_predictions_{case_id[:8]}.csv",
    mime="text/csv"
)

# Exportar dados brutos também
csv_raw = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar Dados Brutos",
    data=csv_raw,
    file_name=f"n_of_1_raw_data_{case_id[:8]}.csv",
    mime="text/csv"
)

st.success("✅ Dashboard carregado com sucesso!")

# ==========================================================
# Footer
# ==========================================================
st.divider()
st.caption("N-of-1 Oncology Dashboard | Powered by Lovable Cloud")
