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
    
    # Standardizing and Encoding
    scaler = StandardScaler()
    df["dmsp_scaled"] = scaler.fit_transform(df[["dmsp_total_light"]])
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    df["urban_dummy"] = encoder.fit_transform(df[["area_type"]])[:, 0]
    
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Maps", "Data", "Model", "SHAP Analysis"])

# 📊 Data Page
if page == "Data":
    st.title("Data Exploration")
    st.markdown("### Explore Household Consumption & Nightlight Data")
    
    # Area Type Filter
    area_filter = st.selectbox("Select Area Type", ["All", "Urban", "Rural"])
    df_filtered = df if area_filter == "All" else df[df["area_type"] == area_filter.upper()]
    
    # Overview
    st.subheader("Dataset Overview")
    st.write(df_filtered.describe())
    
    # Consumption Distribution
    st.subheader("Household Consumption Distribution")
    fig_hist = px.histogram(df_filtered, x="secc_cons", nbins=50, marginal="violin", opacity=0.75,
                            color_discrete_sequence=["royalblue"], title="Household Consumption Expenditure Distribution")
    st.plotly_chart(fig_hist)
    
    # Nightlight vs Consumption
    st.subheader("Nightlight Intensity vs Consumption")
    fig_scatter = px.scatter(df_filtered, x="dmsp_total_light", y="secc_cons", color="area_type",
                             trendline="ols", title="Nightlight Intensity vs Household Consumption")
    st.plotly_chart(fig_scatter)
    
    # Urban vs Rural Comparison
    st.subheader("Urban vs Rural Consumption")
    fig_box = px.box(df, x="area_type", y="secc_cons", title="Urban vs Rural Household Consumption")
    st.plotly_chart(fig_box)
    
    # Feature Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = df_filtered[["log_secc_cons", "dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]].corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", showscale=True
    )
    st.plotly_chart(fig_heatmap)
    
    # Insights
    st.markdown("### Insights:")
    st.markdown("- **Nightlight Intensity**: Higher values indicate more urbanized areas.")
    st.markdown("- **Household Consumption**: Log transformation helps analyze patterns.")
    st.markdown("- **Urban Dummy & Interaction Term**: Captures differences in urban and rural settings.")

# 🤖 Model Page
elif page == "Model":
    st.title("Neural Network Model for Prediction")
    
    # Define features and target variable
    features = ["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]
    target = "log_secc_cons"
    
    # Convert to NumPy arrays
    X = df[features].values
    y = df[target].values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown(f"Training Set: X_train: {X_train.shape}, y_train: {y_train.shape}")
    st.markdown(f"Testing Set: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    st.markdown("**Why 80-20 Split?**")
    st.markdown("- 80% of the data is used for training so the model can learn patterns.")
    st.markdown("- 20% is reserved for testing to evaluate how well the model generalizes to unseen data.")

# 🗺️ Maps Page
elif page == "Maps":
    st.title("Nightlight and Consumption Inequality Maps")

    # File Paths
    map1_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/05_reports/maps/cons_ineq.png"
    map2_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/05_reports/maps/nightlights.png"

    # Function to load & resize images
    def load_and_resize(image_path, width=600, height=600):
        image = Image.open(image_path)
        return image.resize((width, height))

    # Resize images
    map1 = load_and_resize(map1_path)
    map2 = load_and_resize(map2_path)

    # Create columns for side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(map1, caption="Consumption Inequality", use_container_width=True)

    with col2:
        st.image(map2, caption="Nightlight Intensity", use_container_width=True)

# 🔍 SHAP Analysis Page
elif page == "SHAP Analysis":
    st.title("SHAP Analysis Coming Soon")
    st.markdown("This section will contain SHAP value interpretations for feature importance analysis.")

# 🔍 Acknowledgements Page
elif page == "Acknowledgements":
    st.title("Acknowledgements")
    st.markdown("And a huge thank you to ... again, this will be updated!")
