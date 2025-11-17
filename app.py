import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from supabase import create_client, Client

# ==========================================================
# Conexão com Supabase
# ==========================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="N-of-1 Oncology Dashboard", layout="wide")

st.title("🧬 N-of-1 Oncology Dashboard – Tumor Burden Predictor")
st.markdown("""
Este dashboard lê automaticamente os dados do paciente direto do Supabase,
sem necessidade de upload de CSV.
""")


# ==========================================================
# 1. Seleção do Caso + Carregamento
# ==========================================================
st.header("📂 1. Selecione o Caso")

case_id = st.text_input("Digite o case_id do paciente:", placeholder="ex: d1504f59-xxxx-xxxx")

if not case_id:
    st.stop()

# Buscar timepoints na tabela
query = (
    supabase.table("timepoints")
    .select("*")
    .eq("case_id", case_id)
    .order("date", desc=False)
)

response = query.execute()

if len(response.data) == 0:
    st.error("Nenhum timepoint encontrado para esse case_id.")
    st.stop()

df = pd.DataFrame(response.data)

# Converter colunas
df["date"] = pd.to_datetime(df["date"])
df["sum_mm"] = pd.to_numeric(df["sum_mm"], errors="coerce")

st.write("### 🔍 Timepoints carregados:")
st.dataframe(df)


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

st.write(f"**Slope (mm/dia):** `{slope:.4f}`")
st.write(f"**Intercept:** `{intercept:.2f}`")

future_days = st.multiselect(
    "Selecione horizontes de previsão (em dias):",
    [30, 60, 90, 120, 180, 240, 360],
    default=[30, 90, 180]
)

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

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["date"], df["sum_mm"], marker="o", linewidth=3, label="Real")
ax.plot(pred_df["date"], pred_df["predicted_sum_mm"], marker="x", linestyle="--", linewidth=2, label="Predito")

ax.set_xlabel("Data")
ax.set_ylabel("Tumor burden (mm)")
ax.set_title("N-of-1 – Evolução do Tumor")
ax.grid(True)
ax.legend()

st.pyplot(fig)


# ==========================================================
# 4. Probabilidade heurística de progressão
# ==========================================================
st.header("📉 4. Probabilidade de Progressão")

prob_pd = float(np.clip((slope / 1.0) * 100 + 50, 0, 100))

st.metric(
    label="Probabilidade aproximada de progressão nos próximos 6 meses",
    value=f"{prob_pd:.1f}%"
)

st.caption("""
*Obs.: baseado no slope atual; para probabilidade real use modelo Bayesiano.*
""")


# ==========================================================
# 5. Exportação
# ==========================================================
st.header("📤 5. Exportar Previsões")

csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar CSV",
    data=csv,
    file_name="n_of_1_predictions.csv",
    mime="text/csv"
)

st.success("Dashboard carregado com sucesso!")
