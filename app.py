import streamlit as st
import pandas as pd
import os
from pipeline.pipeline import DataPipeline

st.set_page_config(page_title="Playstore Dashboard", layout="wide")

st.title("📊 Google Playstore Dashboard")

DATA_PATH = "data/raw/googleplaystore.csv"
OUTPUT_PATH = "data/processed"

# Run pipeline
if st.button("Run Pipeline"):
    pipeline = DataPipeline(DATA_PATH, OUTPUT_PATH)
    df = pipeline.run()
    st.success("Pipeline executed successfully!")

# Load processed data
processed_file = os.path.join(OUTPUT_PATH, "processed.csv")

if os.path.exists(processed_file):
    df = pd.read_csv(processed_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Stats")
    st.write(df.describe())

    st.subheader("Category Distribution")
    st.bar_chart(df['Category'].value_counts())

# Show reports (your existing images)
st.subheader("Pre-generated Reports")

report_dir = "reports"

for file in os.listdir(report_dir):
    if file.endswith(".png"):
        st.image(os.path.join(report_dir, file), caption=file)