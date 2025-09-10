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

    # Show summary statistics
    st.write("### Summary Statistics")
    st.write(data.describe())

    
   # Correlation heatmap
st.write("### Correlation Heatmap")

numeric_data = data.select_dtypes(include=[np.number])  # only numeric columns

if numeric_data.shape[1] > 1:  # check if at least 2 numeric columns
    fig, ax = plt.subplots()
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ Not enough numeric columns for correlation analysis.")


    # Suggestions based on data
    st.write("### Suggestions")
    if data.corr().abs().max().max() > 0.8:
        st.success("⚡ Strong correlation found in your dataset. Regression might be useful.")
    else:
        st.info("ℹ️ No strong correlation detected. Try exploring distributions or clustering.")
