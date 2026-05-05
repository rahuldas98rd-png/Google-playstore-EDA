import streamlit as st
import pandas as pd
import os
import joblib
from pipeline.pipeline import DataPipeline

st.set_page_config(layout="wide")
st.title("📊 Playstore Analytics Dashboard")

# ==============================
# Paths
# ==============================
DATA_PATH = "data/raw/googleplaystore.csv"
OUTPUT_PATH = "data/processed"

CLEAN_FILE = os.path.join(OUTPUT_PATH, "clean.csv")
PROCESSED_FILE = os.path.join(OUTPUT_PATH, "processed.csv")

MODEL_DIR = "artifacts/models"
REPORT_DIR = "artifacts/reports"

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("Run Pipeline"):
    pipeline = DataPipeline(DATA_PATH, OUTPUT_PATH)
    pipeline.run()
    st.sidebar.success("Pipeline executed successfully!")

# ==============================
# Caching
# ==============================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(MODEL_DIR, "rating_model.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, "success_model.pkl"))
    return reg, clf

# ==============================
# Load Data
# ==============================
if not os.path.exists(CLEAN_FILE):
    st.warning("⚠️ Please run the pipeline first.")
    st.stop()

df_clean = load_data(CLEAN_FILE)

# ==============================
# Sidebar Filters (SAFE)
# ==============================
if 'Category' in df_clean.columns:
    category = st.sidebar.selectbox(
        "Category",
        sorted(df_clean['Category'].dropna().unique())
    )
    filtered_df = df_clean[df_clean['Category'] == category]
else:
    st.error("❌ 'Category' column missing in clean data.")
    st.stop()

# ==============================
# KPIs
# ==============================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Apps", len(filtered_df))

col2.metric(
    "Avg Rating",
    round(filtered_df['Rating'].mean(), 2)
    if 'Rating' in filtered_df else "N/A"
)

col3.metric(
    "Total Installs",
    int(filtered_df['Installs'].sum())
    if 'Installs' in filtered_df else "N/A"
)

# ==============================
# Data Preview
# ==============================
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df.head(50))

# ==============================
# Charts
# ==============================
st.subheader("📊 Installs Distribution")

if 'Installs' in filtered_df.columns:
    st.bar_chart(filtered_df['Installs'].head(20))
else:
    st.warning("Installs column not found.")

# ==============================
# Prediction Section
# ==============================
st.subheader("🔮 Predict App Performance")

reg_model_path = os.path.join(MODEL_DIR, "rating_model.pkl")
clf_model_path = os.path.join(MODEL_DIR, "success_model.pkl")

if os.path.exists(reg_model_path) and os.path.exists(clf_model_path):

    reviews = st.number_input("Reviews", value=1000, min_value=0)
    installs = st.number_input("Installs", value=50000, min_value=0)

    if st.button("Predict"):
        reg_model, clf_model = load_models()

        pred_rating = reg_model.predict([[reviews, installs]])[0]
        pred_success = clf_model.predict([[reviews, installs]])[0]

        st.success(f"⭐ Predicted Rating: {round(pred_rating, 2)}")
        if pred_success:
            st.success("🚀 Success")
        else:
            st.warning("⚠️ Failure")

else:
    st.info("ℹ️ Train models by running the pipeline.")

# ==============================
# EDA Reports
# ==============================
st.subheader("📸 EDA Reports")

if os.path.exists(REPORT_DIR):
    report_files = [f for f in os.listdir(REPORT_DIR) if f.endswith(".png")]

    if report_files:
        for file in report_files:
            st.image(os.path.join(REPORT_DIR, file), caption=file)
    else:
        st.info("No reports found.")
else:
    st.info("Report directory not found.")