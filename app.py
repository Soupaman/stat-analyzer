import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.title("📊 Smart Statistical Analyzer")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    # Summary statistics
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Correlation heatmap (numeric columns only)
    st.write("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] > 1:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Not enough numeric columns for correlation analysis.")

    # Suggestions
    st.write("### Suggestions")
    if numeric_data.shape[1] > 1 and numeric_data.corr().abs().max().max() > 0.8:
        st.success("⚡ Strong correlation found in your dataset. Regression might be useful.")
    else:
        st.info("ℹ️ No strong correlation detected. Try exploring distributions or clustering.")

