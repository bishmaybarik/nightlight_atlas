import streamlit as st

# Streamlit page config
st.set_page_config(page_title="Nightlight Atlas Dashboard", layout="wide")

# Title of the dashboard
st.title("Nightlight Atlas Dashboard")

# Sidebar for map selection
map_option = st.sidebar.radio("Select Map:", ("Nightlights Map", "Consumption Inequality Map"))

# Define file paths
nightlights_path = "./assets/nightlights.html"
cons_ineq_path = "./assets/cons_ineq.html"

# Display the selected map
if map_option == "Nightlights Map":
    st.components.v1.html(open(nightlights_path, 'r', encoding='utf-8').read(), height=600)
elif map_option == "Consumption Inequality Map":
    st.components.v1.html(open(cons_ineq_path, 'r', encoding='utf-8').read(), height=600)

# Run this using: streamlit run your_script.py
