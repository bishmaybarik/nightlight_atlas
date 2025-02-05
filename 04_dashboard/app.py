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
import tempfile


# File Paths
# Define a global variable for the base directory
BASE_DIR = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas"
# BASE_DIR = "https://github.com/bishmaybarik/nightlight_atlas/main"

# Use it to define file paths
data_path = os.path.join(BASE_DIR, "01_data", "02_processed", "secc_combined_updated.parquet")
save_path = os.path.join(BASE_DIR, "05_reports", "model_results")

# os.makedirs(save_path, exist_ok=True)

# Use a temporary directory
save_path = tempfile.mkdtemp()

# Now, use this path to save files
print(f"Files will be saved at: {save_path}")


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
page = st.sidebar.radio("Go to", ["Maps", "Data", "Model", "SHAP Analysis", "Acknowledgements"])

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
    
    st.markdown("## Overview")
    st.markdown(
        "This neural network model predicts **household consumption expenditure** "
        "using features such as nightlight intensity and urban classification."
    )
    
    # Define features and target variable
    features = ["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]
    target = "log_secc_cons"

    st.markdown("### Features and Target")
    st.markdown("- **dmsp_scaled**: Scaled nightlight intensity.")
    st.markdown("- **urban_dummy**: A binary variable indicating urban (1) or rural (0) areas.")
    st.markdown("- **nightlight_area_interaction**: Interaction term capturing urban-rural differences in nightlight influence.")
    st.markdown("- **Target**: `log_secc_cons` (log-transformed consumption expenditure).")

    # Convert to NumPy arrays
    X = df[features].values
    y = df[target].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown(f"**Training Set:** X_train: `{X_train.shape}`, y_train: `{y_train.shape}`")
    st.markdown(f"**Testing Set:** X_test: `{X_test.shape}`, y_test: `{y_test.shape}`")

    st.markdown("### Why an 80-20 Split")
    st.markdown("- **80% for Training:** The model learns patterns from a larger dataset.")
    st.markdown("- **20% for Testing:** Ensures we can evaluate how well the model generalizes to unseen data.")

    # Model Architecture
    st.markdown("## Model Architecture")
    st.markdown("We use a feedforward neural network with two hidden layers:")
    
    st.code(
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),  # First hidden layer
            tf.keras.layers.Dense(32, activation="relu"),                   # Second hidden layer
            tf.keras.layers.Dense(1)                                         # Output layer
        ])
        """,
        language="python"
    )

    st.markdown("- **Hidden Layers**: Use ReLU activation to introduce non-linearity.")
    st.markdown("- **Output Layer**: Produces a single predicted value for log consumption.")

    # Model Compilation
    st.markdown("### Compiling the Model")
    st.markdown("We use Adam optimizer and Mean Squared Error (MSE) as the loss function.")

    st.code(
        """
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        """,
        language="python"
    )

    # Training Process
    st.markdown("## Model Training")
    st.markdown(
        "The model is trained over 50 epochs with a batch size of 64. "
        "We validate performance using the test set."
    )

    st.code(
        """
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
        """,
        language="python"
    )

    # Model Evaluation
    st.markdown("## Model Evaluation")
    st.markdown(
        "After training, we evaluate the model using Mean Squared Error (MSE) "
        "and Mean Absolute Error (MAE) on the test set."
    )

    st.code(
        """
        test_loss, test_mae = model.evaluate(X_test, y_test)
        """,
        language="python"
    )

    # Making Predictions
    st.markdown("## Making Predictions")
    st.markdown("We transform the predictions back from log scale to actual consumption values.")

    st.code(
        """
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)      # Convert predictions back from log scale
        y_test_actual = np.expm1(y_test)   # Convert actual values back from log scale
        """,
        language="python"
    )

    # Scatter Plot of Predictions vs. Actual Values
    st.markdown("### Visualizing Predictions")
    st.markdown("A scatter plot compares actual vs. predicted household consumption.")

    st.code(
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=y_test_actual, y=y_pred.flatten(), alpha=0.5)
        plt.xlabel("Actual Household Consumption")
        plt.ylabel("Predicted Household Consumption")
        plt.title("Actual vs Predicted Household Consumption")
        plt.show()
        """,
        language="python"
    )

    st.markdown("## Results of the Model Tested")

    # Load and display the images
    st.markdown("### Training vs Validation Loss")
    training_loss_image = Image.open(os.path.join(BASE_DIR, "05_reports", "model_results", "training_vs_validation_loss.png"))

    st.markdown(
        """
        **Interpretation:**
        - The training loss decreases steadily during the initial epochs, showing that the model learns effectively.
        - The validation loss follows a similar trend but plateaus earlier, suggesting the model may begin to overfit after a certain point.
        - The final losses are close, indicating the model generalizes well on unseen data.
        """
    )

    st.markdown("### Predictions vs Actual Values")
    prediction_image = Image.open(os.path.join(BASE_DIR, "05_reports", "model_results", "actual_vs_predicted_consumption.png"))
    st.image(prediction_image, caption="Predicted vs Actual Household Consumption")

    st.markdown(
        """
        **Interpretation:**
        - The scatter plot shows predicted values closely aligning with actual values, as most points are near the diagonal.
        - Some variance is observed for extremely high consumption values, which may indicate potential outliers or areas where the model's predictions are less accurate.
        - Overall, the model performs well in capturing the relationship between the features and target variable.
        """
    )

    st.markdown("### Summary of Results")
    st.markdown(
        """
        - The neural network effectively minimizes loss during training and achieves good performance on validation data.
        - The scatter plot highlights strong predictive capability, with minor deviations in edge cases.
        - These results suggest the model is suitable for predicting log-transformed household consumption expenditure.
        """
    )


# 🗺️ Maps Page
elif page == "Maps":
    st.title("Nightlight and Consumption Inequality Maps")

    # File Paths
    map1_path = os.path.join(BASE_DIR, "05_reports", "maps", "cons_ineq.png")
    map2_path = os.path.join(BASE_DIR, "05_reports", "maps", "nightlights.png")

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

