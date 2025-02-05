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
from PIL import Image  # Importing Pillow for image resizing
import requests
import tempfile
import io

# File Paths
BASE_DIR = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main"

# Use a temporary directory
save_path = tempfile.mkdtemp()
print(f"Files will be saved at: {save_path}")

# Function to load data from GitHub
@st.cache_data
def load_data():
    DATA_URL = f"{BASE_DIR}/01_data/02_processed/secc_combined_updated.parquet"
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        df = pd.read_parquet(io.BytesIO(response.content))
        df["area_type"] = df["area_type"].str.strip().str.upper()
        df["log_secc_cons"] = np.log1p(df["secc_cons"])
        df["nightlight_area_interaction"] = df["dmsp_total_light"] * (df["area_type"] == "URBAN").astype(int)
        scaler = StandardScaler()
        df["dmsp_scaled"] = scaler.fit_transform(df[["dmsp_total_light"]])
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        df["urban_dummy"] = encoder.fit_transform(df[["area_type"]])[:, 0]
        return df
    else:
        st.error("Failed to load dataset. Please check the GitHub link.")
        return None

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Maps", "Data", "Model", "SHAP Analysis", "Acknowledgements"])

# 📊 Data Page
if page == "Data":
    st.title("Data Exploration")
    area_filter = st.selectbox("Select Area Type", ["All", "Urban", "Rural"])
    df_filtered = df if area_filter == "All" else df[df["area_type"] == area_filter.upper()]
    st.subheader("Dataset Overview")
    st.write(df_filtered.describe())
    
    st.subheader("Household Consumption Distribution")
    fig_hist = px.histogram(df_filtered, x="secc_cons", nbins=50, marginal="violin", opacity=0.75)
    st.plotly_chart(fig_hist)
    
    st.subheader("Nightlight Intensity vs Consumption")
    try:
        import statsmodels.api as sm  # Ensure statsmodels is installed
        fig_scatter = px.scatter(df_filtered, x="dmsp_total_light", y="secc_cons", color="area_type", trendline="ols")
        st.plotly_chart(fig_scatter)
    except ModuleNotFoundError:
        st.error("Missing dependency: statsmodels. Install it using `pip install statsmodels`.")

    st.subheader("Urban vs Rural Consumption")
    fig_box = px.box(df, x="area_type", y="secc_cons")
    st.plotly_chart(fig_box)

# 🗺️ Maps Page
elif page == "Maps":
    st.title("Nightlight and Consumption Inequality Maps")
    
    def load_image_from_url(image_url):
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error(f"Failed to load image: {image_url}")
            return None

    map1_url = f"{BASE_DIR}/05_reports/maps/cons_ineq.png"
    map2_url = f"{BASE_DIR}/05_reports/maps/nightlights.png"

    map1 = load_image_from_url(map1_url)
    map2 = load_image_from_url(map2_url)

    col1, col2 = st.columns(2)
    with col1:
        if map1:
            st.image(map1, caption="Consumption Inequality", use_container_width=True)
    with col2:
        if map2:
            st.image(map2, caption="Nightlight Intensity", use_container_width=True)

# 🤖 Model Page
elif page == "Model":
    st.title("Neural Network Model for Prediction")
    features = ["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]
    target = "log_secc_cons"
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown("### Model Architecture")
    st.code(
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        """,
        language="python"
    )

    # Display model results images
    def load_model_image(image_url):
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    
    loss_image_url = f"{BASE_DIR}/05_reports/model_results/training_vs_validation_loss.png"
    prediction_image_url = f"{BASE_DIR}/05_reports/model_results/actual_vs_predicted_consumption.png"
    
    loss_image = load_model_image(loss_image_url)
    prediction_image = load_model_image(prediction_image_url)
    
    if loss_image:
        st.image(loss_image, caption="Training vs Validation Loss")
    else:
        st.error("Failed to load training loss image.")
    
    if prediction_image:
        st.image(prediction_image, caption="Predicted vs Actual Household Consumption")
    else:
        st.error("Failed to load predictions image.")

# 🔍 SHAP Analysis Page
elif page == "SHAP Analysis":
    st.title("SHAP Analysis Coming Soon")
    
# 🔍 Acknowledgements Page
elif page == "Acknowledgements":
    st.title("Acknowledgements")
