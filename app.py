import streamlit as st
import pandas as pd
import os
import joblib
from pipeline.pipeline import DataPipeline

st.set_page_config(layout="wide")

st.title("📊 Playstore Analytics Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

DATA_PATH = "data/raw/googleplaystore.csv"
OUTPUT_PATH = "data/processed"

if st.sidebar.button("Run Pipeline"):
    pipeline = DataPipeline(DATA_PATH, OUTPUT_PATH)
    df = pipeline.run()
    st.sidebar.success("Pipeline executed")

# Load data
file_path = os.path.join(OUTPUT_PATH, "processed.csv")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Sidebar filters
    category = st.sidebar.selectbox("Category", df['Category'].unique())
    filtered_df = df[df['Category'] == category]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Apps", len(filtered_df))
    col2.metric("Avg Rating", round(filtered_df['Rating'].mean(), 2))
    col3.metric("Total Installs", int(filtered_df['Installs'].sum()))

    st.subheader("Filtered Data")
    st.dataframe(filtered_df.head())

    # Charts
    st.subheader("Installs Distribution")
    st.bar_chart(filtered_df['Installs'].head(20))

    # Load models
    reg_model_path = "artifacts/models/rating_model.pkl"
    clf_model_path = "artifacts/models/success_model.pkl"

    if os.path.exists(reg_model_path):
        st.subheader("🔮 Prediction")

        reviews = st.number_input("Reviews", value=1000)
        installs = st.number_input("Installs", value=50000)

        if st.button("Predict"):
            reg_model = joblib.load(reg_model_path)
            clf_model = joblib.load(clf_model_path)

            pred_rating = reg_model.predict([[reviews, installs]])[0]
            pred_success = clf_model.predict([[reviews, installs]])[0]

            st.write(f"Predicted Rating: {round(pred_rating,2)}")
            st.write(f"Success: {'Yes' if pred_success else 'No'}")

    # Show reports
    st.subheader("📸 EDA Reports")
    report_dir = "artifacts/reports"

    for file in os.listdir(report_dir):
        if file.endswith(".png"):
            st.image(os.path.join(report_dir, file))