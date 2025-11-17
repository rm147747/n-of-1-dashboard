import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================
# CONFIGURAÇÃO DO APP
# ==========================
st.set_page_config(
    page_title="N-of-1 Oncology Dashboard",
    layout="wide"
)

# ==========================
# TÍTULO E INTRODUÇÃO
# ==========================
st.title("🧬 N-of-1 Oncology Dashboard – Tumor Burden Predictor")

st.markdown("""
Este dashboard permite acompanhar a evolução individual do tumor (soma dos diâmetros), 
calcular slopes de resposta/progressão e gerar previsões personalizadas para os próximos meses
baseadas exclusivamente no comportamento individual de cada paciente.
""")

# ==========================
# 1 — CARREGAMENTO DOS DADOS
# ==========================
st.header("📂 1. Upload dos Dados do Paciente")

uploaded_file = st.file_uploader(
    "Envie um CSV com as colunas: date, sum_mm (formato mm de tumor burden)",
    type=["csv"]
)

# Dados default de exemplo: paciente com evolução real
default_data = {
    "date": ["2022-03-17","2022-06-27","2022-08-03","2023-07-27","2025-11-04"],
    "sum_mm": [182, 79, 79, 520, 510]
}

df_default = pd.DataFrame(default_data)
df_default["date"] = pd.to_datetime(df_default["date"])
df_default["days"] = (df_default["date"] - df_default["date"].min()).dt.days

# Se o usuário fizer upload, usar o CSV. Se não, usar default.
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])
else:
    df = df_default.copy()

df = df.sort_values("date")
T0 = df["date"].min()
df["days"] = (df["date"] - T0).dt.days

st.write("### 🔍 Dados carregados:")
st.dataframe(df)

# ================================
# 2 — MODELAGEM PREDITIVA N-of-1
# ================================
st.header("🧠 2. Modelagem Preditiva (Linear Regression N-of-1)")

X = df["days"].values.reshape(-1,1)
y = df["sum_mm"].values

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

st.write(f"**Slope calculado (mm/dia):** `{slope:.4f}`")
st.write(f"**Intercepto:** `{intercept:.2f}`")

# ================================
# 3 — PREVISÕES FUTURAS
# ================================
st.header("📈 3. Previsões Futuras")

future_days = st.multiselect(
    "Selecione os horizontes de previsão (em dias):",
    [30, 60, 90, 120, 180, 240, 360],
    default=[30, 90, 180]
)

preds = {
    "days_from_t0": future_days,
    "predicted_sum_mm": [intercept + slope*d for d in future_days],
    "date": [(T0 + timedelta(days=d)).strftime("%Y-%m-%d") for d in future_days]
}

pred_df = pd.DataFrame(preds)

st.write("### 📊 Tabela de Previsões")
st.table(pred_df)

# ================================
# 4 — GRÁFICO REAL VS PREDITO
# ================================
st.header("📉 4. Curva Temporal: Real vs Predito")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(df["date"], df["sum_mm"], marker='o', linewidth=3, label="Real")
ax.plot(pred_df["date"], pred_df["predicted_sum_mm"], marker='x', linestyle='--',
        linewidth=2, label="Predito")

ax.set_xlabel("Data")
ax.set_ylabel("Tumor burden (mm)")
ax.set_title("Evolução do Tumor – N-of-1")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# ================================
# 5 — PROBABILIDADE HEURÍSTICA DE PROGRESSÃO
# ================================
st.header("📉 5. Probabilidade Aproximada de Progressão")

prob_pd = np.clip((slope / 1.0) * 100 + 50, 0, 100)

st.metric(
    label="Probabilidade aproximada de progressão nos próximos 6 meses",
    value=f"{prob_pd:.1f} %"
)

st.caption("""
🔬 *Este cálculo é heurístico e baseado apenas no comportamento N-of-1 do paciente.  
Para probabilidades reais e clínicas, recomenda-se modelagem Bayesiana (PyMC).*  
""")

# ================================
# 6 — EXPORTAÇÃO
# ================================
st.header("📤 6. Exportar Previsões")

csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar CSV com previsões",
    data=csv,
    file_name="n_of_1_predictions.csv",
    mime="text/csv"
)

st.success("Dashboard carregado com sucesso! 🚀")
