# -*- coding: utf-8 -*-
"""
Enhanced Global Development Clustering Application
"""

import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Load dataset and model
df = pd.read_excel("Data.xlsx")
with open('kmeans.pkl', 'rb') as load:
    model = pickle.load(load)

# Select the important columns for prediction
important_features = [
    "Birth Rate", "CO2 Emissions", "Energy Usage", "GDP", 
    "Health Exp % GDP", "Infant Mortality Rate", "Internet Usage", 
    "Life Expectancy Female", "Life Expectancy Male", "Mobile Phone Usage", 
    "Population 0-14", "Population 15-64", "Population 65+", 
    "Population Total", "Population Urban", "Tourism Inbound"
]

def predict_cluster(input_data):
    try:
        sanitized_data = []
        for value in input_data:
            # Handle non-numeric inputs and percentage signs
            if isinstance(value, str):
                value = value.replace('%', '').strip()  # Remove percentage and strip spaces
            try:
                sanitized_data.append(float(value))  # Convert to float
            except ValueError:
                sanitized_data.append(0.0)  # Default to 0.0 for invalid entries
        prediction = model.predict([sanitized_data])[0]
        return prediction
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Define the main application
def main():
    st.set_page_config(page_title="Global Development Clustering", layout="wide")

    st.title("Global Development Clustering Application")
    st.markdown(
        """
        This application clusters countries based on key development metrics using Machine Learning.
        You can input values to predict the cluster for a given set of metrics and explore data visualizations.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Enter The Details")
    input_data = [st.sidebar.text_input(f, "0") for f in important_features]

    # Predict button
    if st.sidebar.button("Predict Cluster"):
        cluster = predict_cluster(input_data)
        st.sidebar.success(f"Predicted Cluster: {cluster}")

    # Data visualization section
    st.header("Explore the Dataset")
    st.write("### Raw Dataset")
    st.dataframe(df[important_features].head())

    # Visualization options
    st.write("### Visualizations")
    viz_choice = st.selectbox(
        "Select a Visualization:", 
        ["Correlation Heatmap", "Scatter Plot", "Bar Chart", "Map", "Cluster Visualization"]
    )

    if viz_choice == "Correlation Heatmap":
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[important_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif viz_choice == "Scatter Plot":
        x_axis = st.selectbox("X-axis Feature:", important_features)
        y_axis = st.selectbox("Y-axis Feature:", important_features)
        scatter_fig = px.scatter(df, x=x_axis, y=y_axis, color=important_features[-1])
        st.plotly_chart(scatter_fig)

    elif viz_choice == "Bar Chart":
        bar_feature = st.selectbox("Feature for Bar Chart:", important_features)
        bar_fig = px.bar(df, x=bar_feature, y=important_features[-1])
        st.plotly_chart(bar_fig)

    elif viz_choice == "Map":
        st.write("#### Map of Countries in Dataset")
        if "Country" in df.columns and "Population Total" in df.columns:
            map_fig = px.choropleth(
                df, locations="Country", locationmode="country names",
                color="Population Total", hover_name="Country",
                title="World Map of Population"
            )
            st.plotly_chart(map_fig)
        else:
            st.error("Map visualization requires 'Country' and 'Population Total' columns.")

    elif viz_choice == "Cluster Visualization":
        st.write("#### Cluster Visualization")
        if "Country" in df.columns:
            df["Cluster"] = model.predict(df[important_features])
            cluster_fig = px.scatter(
                df, x="GDP", y="Life Expectancy Female", color="Cluster", 
                hover_name="Country", title="Cluster Visualization: GDP vs Life Expectancy Female"
            )
            st.plotly_chart(cluster_fig)
        else:
            st.error("Cluster visualization requires 'Country' column.")

    # Footer
    st.markdown("---")
    st.text("Developed by Group 1:Ubed M A,Spoorthi Prasad, Kavya Javai, NandakiShore Boini, Preethi, Amit Chougule, Ganesh Ghanwat ")

# Run the application
if __name__ == "__main__":
    main()
