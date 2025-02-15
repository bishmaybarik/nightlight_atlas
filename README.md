# Nightlight Atlas: Predicting Household Consumption using Nightlights

## Overview

This project explores the relationship between **nightlight intensity** and **household consumption expenditure** in India. Using high-resolution geospatial data and machine learning techniques, I build a predictive model that estimates household consumption from satellite-recorded nightlight intensity and other regional characteristics.

In the later part of the project, especially once the analysis is complete, I have also developed a web app — a Dashboard using Streamlit — whose codes can be found in `04_dashboard`. The web app provides an interactive visualization of the analysis and can be accessed at [Nightlight Atlas](https://nightlight-atlas.streamlit.app/).

## Key Features
- **Dataset**: The project leverages **SECC (2011) consumption data** and **DMSP/VIIRS nightlight data**.
- **Machine Learning Model**: A **Neural Network** is trained to predict household consumption based on nightlight intensity and urban-rural classification.
- **Data Visualization**: Interactive dashboards and maps illustrate spatial variations in nightlights and economic inequality.
- **SHAP Analysis (Coming Soon)**: I will analyze feature importance using SHAP values to interpret the model’s predictions.

## Directory Structure
```
.
├── 01_data
│   ├── 01_raw
│   │   ├── shrug-con-keys-csv
│   │   ├── shrug-dmsp-csv
│   │   ├── shrug-secc-cons-rural-csv
│   │   ├── shrug-secc-cons-urban-csv
│   │   └── shrug-viirs-annual-csv
│   ├── 02_processed
│   └── 03_shapefiles
│       └── shrug-shrid-poly-shp
├── 02_notebooks
├── 03_src
│   └── __pycache__
├── 04_dashboard
│   ├── assets
│   └── components
└── 05_reports
    ├── data
    ├── maps
    └── model_results
```

## Data Used

The primary dataset comes from the **Socio-Economic and Caste Census (SECC) 2011**, combined with nightlight intensity data from **DMSP-OLS and VIIRS** satellites.

### Data Preprocessing
- **Log transformation**: `log_secc_cons = log(1 + secc_cons)`
- **Feature Engineering**:
  - `nightlight_area_interaction`: Interaction term between nightlight intensity and urban classification.
  - `dmsp_scaled`: Standardized nightlight intensity.
  - `urban_dummy`: One-hot encoded urban-rural indicator.

## Visualizing Spatial Inequality

<div style="display: flex; justify-content: space-between;">
    <img src="https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/cons_ineq.png" style="width: auto; height: 400px;" alt="Consumption Inequality Across India">
    <img src="https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/maps/nightlights.png" style="width: auto; height: 400px;" alt="Nightlight Intensity Across India">
</div>



## Neural Network Model

I use a **feedforward neural network** with two hidden layers to predict household consumption expenditure:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

### Model Training
- **Train-Test Split**: 80% training, 20% testing
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 50
- **Batch Size**: 64

```python
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
```

### Training Performance
![Training vs Validation Loss](https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/training_vs_validation_loss.png)

### Model Predictions

To evaluate the model, I compare the predicted and actual household consumption values.

```python
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)
```

### Actual vs Predicted Household Consumption
![Actual vs Predicted](https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/model_results/actual_vs_predicted_consumption.png)

```markdown
# SHAP Analysis

This section provides an analysis of SHAP (SHapley Additive exPlanation) values to interpret feature importance in the neural network model. Below, we include key SHAP visualizations to understand how different features impact household consumption expenditure predictions.

## SHAP Summary Plot

![SHAP Summary Plot](https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/SHAP/summary_plot.png)

The SHAP summary plot presents the distribution of SHAP values for each feature. The key takeaways from the plot are:
- **`dmsp_scaled`** has the most significant impact on model predictions, with both positive and negative effects.
- **`nightlight_area_interaction`** and **`urban_dummy`** also contribute to the predictions, but their effects are comparatively smaller.
- The color gradient indicates feature values, where higher values are in pink and lower values are in blue.

## SHAP Dependence Plot

![SHAP Dependence Plot](https://raw.githubusercontent.com/bishmaybarik/nightlight_atlas/main/05_reports/SHAP/dependence_plot.png)

The SHAP dependence plot illustrates the interaction between `dmsp_scaled` and `urban_dummy`. Key observations:
- Higher values of `dmsp_scaled` positively influence the model’s output up to a certain point, after which the impact diminishes.
- The impact varies depending on `urban_dummy`, with urban areas (pink points) showing a slightly different pattern compared to rural areas (blue points).

These insights help in understanding how nightlight intensity (dmsp_scaled) contributes to household consumption expenditure predictions and how it interacts with other features like urbanization.
```

## Acknowledgements

I would like to thank **Development Data Lab** for their outstanding work in compiling these datasets. Proper credits are given to the data contributors below:

### References
- Asher, S., Lunt, T., Matsuura, R., & Novosad, P. (2021). *Development research at high geographic resolution: An analysis of night-lights, firms, and poverty in India using the SHRUG open data platform.* The World Bank Economic Review, 35(4).
- Henderson, J. V., Storeygard, A., & Weil, D. N. (2011). *A bright idea for measuring economic growth.* American Economic Review.

---

**Author**: Bishmay Barik  
**GitHub**: [bishmaybarik](https://github.com/bishmaybarik)
