import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
import io
import os

# Streamlit Page Config
st.set_page_config(
    page_title="Nightlight Atlas Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Nightlight Atlas of Rural Economic Development in India")

# GitHub Raw File URLs
DATA_URL = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/01_data/02_processed/secc_combined_updated.parquet"
MAP1_URL = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/cons_ineq.png"
MAP2_URL = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/nightlights.png"
LOSS_PLOT_URL = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/training_vs_validation_loss.png"
PREDICTION_PLOT_URL = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/actual_vs_predicted_consumption.png"

# Function to load data from GitHub
@st.cache_data
def load_data():
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        df = pd.read_parquet(io.BytesIO(response.content))

        # Feature Engineering
        df["area_type"] = df["area_type"].str.strip().str.upper()
        df["log_secc_cons"] = np.log1p(df["secc_cons"])
        df["nightlight_area_interaction"] = df["dmsp_total_light"] * (df["area_type"] == "URBAN").astype(int)

        # Standardizing & Encoding
        scaler = StandardScaler()
        df["dmsp_scaled"] = scaler.fit_transform(df[["dmsp_total_light"]])

        encoder = OneHotEncoder(sparse_output=False, drop="first")
        df["urban_dummy"] = encoder.fit_transform(df[["area_type"]])[:, 0]

        return df
    else:
        st.error("❌ Failed to load dataset. Please check the GitHub link.")
        return None

df = load_data()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Maps", "Data", "Model", "SHAP Analysis", "Acknowledgements"])

# 📊 Data Page
if page == "Data":
    st.title("📊 Data Exploration")
    st.markdown("### Explore Household Consumption & Nightlight Data")

    if df is not None:
        # Area Type Filter
        area_filter = st.selectbox("Select Area Type", ["All", "Urban", "Rural"])
        df_filtered = df if area_filter == "All" else df[df["area_type"] == area_filter.upper()]

        # Dataset Overview
        st.subheader("🔍 Dataset Overview")
        st.write(df_filtered.describe())

        # Histogram: Household Consumption Distribution
        st.subheader("📈 Household Consumption Distribution")
        fig_hist = px.histogram(df_filtered, x="secc_cons", nbins=50, marginal="violin", opacity=0.75,
                                color_discrete_sequence=["royalblue"],
                                title="Household Consumption Expenditure Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter Plot: Nightlight vs Consumption
        st.subheader("💡 Nightlight Intensity vs Consumption")
        fig_scatter = px.scatter(df_filtered, x="dmsp_total_light", y="secc_cons", color="area_type",
                                 trendline="ols", title="Nightlight Intensity vs Household Consumption")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Feature Correlation Heatmap
        st.subheader("🔬 Feature Correlation Heatmap")
        corr = df_filtered[["log_secc_cons", "dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]].corr()
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", showscale=True
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# 🤖 Model Page
elif page == "Model":
    st.title("🤖 Neural Network Model for Prediction")
    st.markdown("### Predicting Household Consumption Using Nightlight Intensity")

    if df is not None:
        # Features & Target
        features = ["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]
        target = "log_secc_cons"
        X = df[features].values
        y = df[target].values

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Architecture
        st.markdown("### Model Architecture")
        st.code("""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
        """, language="python")

        # Load Model Training Plots
        st.subheader("📉 Training vs Validation Loss")
        st.image(LOSS_PLOT_URL, caption="Training vs Validation Loss", use_column_width=True)

        st.subheader("📊 Predictions vs Actual Values")
        st.image(PREDICTION_PLOT_URL, caption="Predicted vs Actual Household Consumption", use_column_width=True)

# 🗺️ Maps Page
elif page == "Maps":
    st.title("🗺️ Nightlight and Consumption Inequality Maps")

    col1, col2 = st.columns(2)

    with col1:
        st.image(MAP1_URL, caption="Consumption Inequality", use_column_width=True)

    with col2:
        st.image(MAP2_URL, caption="Nightlight Intensity", use_column_width=True)

# 🔍 SHAP Analysis Page
elif page == "SHAP Analysis":
    st.title("🔍 SHAP Analysis")
    st.markdown("Feature importance analysis using SHAP values will be added soon.")

# 🙌 Acknowledgements Page
elif page == "Acknowledgements":
    st.title("📜 Acknowledgements")
    st.markdown("Special thanks to all contributors and data providers!")

st.sidebar.markdown("📌 **Data Source:** [GitHub Repo](https://github.com/bishmaybarik/nightlight_atlas)")
