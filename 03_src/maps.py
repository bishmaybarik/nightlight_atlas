import geopandas as gpd
import pandas as pd
import plotly.express as px

# Paths
shapefile_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/03_shapefiles/shrug-shrid-poly-shp/shrid2_open.shp"
dataset_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/02_processed/secc_combined_updated.csv"
output_path_cons = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/04_dashboard/assets/cons_ineq.html"
output_path_night = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/04_dashboard/assets/nightlights.html"

# Load data
shapefile = gpd.read_file(shapefile_path)
dataset = pd.read_csv(dataset_path)

# Merge datasets
shapefile["shrid2"] = shapefile["shrid2"].astype(str)
dataset["shrid2"] = dataset["shrid2"].astype(str)
merged = shapefile.merge(dataset, on="shrid2")

# Convert to GeoDataFrame
merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry")

# Function to create interactive maps
def create_map(column, title, output_path, color_scale):
    fig = px.choropleth(
        merged_gdf,
        geojson=merged_gdf.geometry,
        locations=merged_gdf.index,
        color=column,
        color_continuous_scale=color_scale,
        title=title,
        hover_data={"shrid2": True, column: True},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_html(output_path)

# Generate maps
create_map("secc_cons", "Consumption Expenditure", output_path_cons, "Blues")
create_map("dmsp_total_light", "Nightlight Intensity", output_path_night, "Cividis")

print("Maps saved successfully!")
