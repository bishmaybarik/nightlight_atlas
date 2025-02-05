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
from PIL import Image
import requests
from io import BytesIO

# GitHub Raw URLs
data_url = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/01_data/02_processed/secc_combined_updated.parquet"
map1_url = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/cons_ineq.png"
map2_url = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/nightlights.png"
training_loss_url = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/training_vs_validation_loss.png"
prediction_url = "https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/actual_vs_predicted_consumption.png"

# Load Data
@st.cache_data
def load_data():
    df = pd.read_parquet(data_url, engine='pyarrow')
    df["area_type"] = df["area_type"].str.strip().str.upper()
    df["log_secc_cons"] = np.log1p(df["secc_cons"])
    df["nightlight_area_interaction"] = df["dmsp_total_light"] * (df["area_type"] == "URBAN").astype(int)
    
    # Standardizing and Encoding
    scaler = StandardScaler()
    df["dmsp_scaled"] = scaler.fit_transform(df[["dmsp_total_light"]])
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    df["urban_dummy"] = encoder.fit_transform(df[["area_type"]])[:, 0]
    
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Maps", "Data", "Model", "SHAP Analysis", "Acknowledgements"])

# Function to load images from GitHub
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Data Page
if page == "Data":
    st.title("Data Exploration")
    st.markdown("### Explore Household Consumption & Nightlight Data")
    
    area_filter = st.selectbox("Select Area Type", ["All", "Urban", "Rural"])
    df_filtered = df if area_filter == "All" else df[df["area_type"] == area_filter.upper()]
    
    st.subheader("Dataset Overview")
    st.write(df_filtered.describe())
    
    st.subheader("Household Consumption Distribution")
    fig_hist = px.histogram(df_filtered, x="secc_cons", nbins=50, marginal="violin", opacity=0.75,
                            color_discrete_sequence=["royalblue"], title="Household Consumption Expenditure Distribution")
    st.plotly_chart(fig_hist)
    
    st.subheader("Nightlight Intensity vs Consumption")
    fig_scatter = px.scatter(df_filtered, x="dmsp_total_light", y="secc_cons", color="area_type",
                             trendline="ols", title="Nightlight Intensity vs Household Consumption")
    st.plotly_chart(fig_scatter)
    
    st.subheader("Urban vs Rural Consumption")
    fig_box = px.box(df, x="area_type", y="secc_cons", title="Urban vs Rural Household Consumption")
    st.plotly_chart(fig_box)
    
    st.subheader("Feature Correlation Heatmap")
    corr = df_filtered[["log_secc_cons", "dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]].corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", showscale=True
    )
    st.plotly_chart(fig_heatmap)

# Model Page
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
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        """,
        language="python"
    )
    
    st.markdown("### Training vs Validation Loss")
    st.image(load_image(training_loss_url), caption="Training vs Validation Loss")
    
    st.markdown("### Predictions vs Actual Values")
    st.image(load_image(prediction_url), caption="Predicted vs Actual Household Consumption")

# Maps Page
elif page == "Maps":
    st.title("Nightlight and Consumption Inequality Maps")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(load_image(map1_url), caption="Consumption Inequality", use_container_width=True)
    with col2:
        st.image(load_image(map2_url), caption="Nightlight Intensity", use_container_width=True)

# SHAP Analysis Page
elif page == "SHAP Analysis":
    st.title("SHAP Analysis Coming Soon")
    st.markdown("This section will contain SHAP value interpretations for feature importance analysis.")

# Acknowledgements Page
elif page == "Acknowledgements":
    st.title("Acknowledgements")
    st.markdown("A huge thank you to everyone involved in this research!")
