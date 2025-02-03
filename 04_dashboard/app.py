import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static

# Set paths to data
shapefile_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/03_shapefiles/shrug-shrid-poly-shp/shrid2_open.shp"
dataset_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/02_processed/secc_combined_updated.csv"

# Load data
@st.cache_data
def load_data():
    shapefile = gpd.read_file(shapefile_path)
    dataset = pd.read_csv(dataset_path)
    shapefile["shrid2"] = shapefile["shrid2"].astype(str)
    dataset["shrid2"] = dataset["shrid2"].astype(str)
    return shapefile.merge(dataset, on="shrid2")

data = load_data()

# Streamlit layout
st.title("Nightlight & Consumption Map")
st.write("Toggle between Consumption Expenditure and Nightlight Intensity.")

# Toggle selection
map_type = st.radio("Select Map Type:", ["Consumption Expenditure", "Nightlight Intensity"])

# Base map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Define function to add data to the map
def add_choropleth(column, cmap):
    folium.Choropleth(
        geo_data=data,
        name=column,
        data=data,
        columns=["shrid2", column],
        key_on="feature.properties.shrid2",
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=column
    ).add_to(m)

# Toggle between the two maps
if map_type == "Consumption Expenditure":
    add_choropleth("secc_cons", "Blues")
else:
    add_choropleth("dmsp_total_light", "YlOrRd")

# Display the map
folium_static(m)
