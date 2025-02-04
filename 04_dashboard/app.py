import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

# File Paths
data_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/02_processed/secc_combined_updated.parquet"
save_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/05_reports/model_results"
os.makedirs(save_path, exist_ok=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet(data_path)
    df["area_type"] = df["area_type"].str.strip().str.upper()
    df["log_secc_cons"] = np.log1p(df["secc_cons"])
    df["nightlight_area_interaction"] = df["dmsp_total_light"] * (df["area_type"] == "URBAN").astype(int)
    scaler = StandardScaler()
    df["dmsp_scaled"] = scaler.fit_transform(df[["dmsp_total_light"]])
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    df["urban_dummy"] = encoder.fit_transform(df[["area_type"]])[:, 0]
    return df

df = load_data()

# Sidebar Controls
st.sidebar.title("Dashboard Controls")
area_filter = st.sidebar.selectbox("Select Area Type", ["All", "Urban", "Rural"])

df_filtered = df if area_filter == "All" else df[df["area_type"] == area_filter.upper()]

# Dashboard Title
st.title("Nightlight & Household Consumption Dashboard")
st.markdown("An interactive analysis of nightlight intensity and household consumption expenditure.")

# Data Overview
st.subheader("Dataset Overview")
st.write(df_filtered.describe())

# Smooth Histogram of Consumption
st.subheader("Household Consumption Distribution")
fig_hist = px.histogram(df_filtered, x="secc_cons", nbins=50, marginal="violin", opacity=0.75,
                        color_discrete_sequence=["royalblue"], title="Household Consumption Expenditure Distribution")
st.plotly_chart(fig_hist)

# Scatter Plot
st.subheader("Nightlight Intensity vs Consumption")
fig_scatter = px.scatter(df_filtered, x="dmsp_total_light", y="secc_cons", color="area_type",
                         trendline="ols", title="Nightlight Intensity vs Household Consumption")
st.plotly_chart(fig_scatter)

# Box Plot
st.subheader("Urban vs Rural Consumption")
fig_box = px.box(df, x="area_type", y="secc_cons", title="Urban vs Rural Household Consumption")
st.plotly_chart(fig_box)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
corr = df_filtered[["log_secc_cons", "dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]].corr()
fig_heatmap = ff.create_annotated_heatmap(
    z=corr.values, x=list(corr.columns), y=list(corr.index),
    colorscale="RdBu", showscale=True
)
st.plotly_chart(fig_heatmap)

st.markdown("### Insights:")
st.markdown("- **Nightlight Intensity**: Higher values indicate more urbanized areas.")
st.markdown("- **Household Consumption**: Log transformation helps analyze patterns.")
st.markdown("- **Urban Dummy & Interaction Term**: Captures differences in urban and rural settings.")

# Machine Learning Section
st.markdown("---")
st.subheader("Neural Network Model for Prediction")

# Define features and target variable
features = ["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]
target = "log_secc_cons"

# Convert to NumPy arrays
X = df[features].values
y = df[target].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display data split details
st.markdown(f"Training Set: X_train: {X_train.shape}, y_train: {y_train.shape}")
st.markdown(f"Testing Set: X_test: {X_test.shape}, y_test: {y_test.shape}")

st.markdown("**Why 80-20 Split?**")
st.markdown("- 80% of the data is used for training so the model can learn patterns.")
st.markdown("- 20% is reserved for testing to evaluate how well the model generalizes to unseen data.")
st.markdown("- Example: If we have 1000 samples, 800 are used for training and 200 for testing.")
