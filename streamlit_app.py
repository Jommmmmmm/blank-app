import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Weather Data Generator & Prediction",
    page_icon="🌦️",
    layout="wide"
)

# Generate Synthetic Weather Data
def generate_weather_data(n_samples, temp_range, humidity_range, wind_range, precip_range):
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data_list = []  # List to hold data for each day
    
    for day_name in days:
        # Generate data for each day
        day_data = pd.DataFrame({
            'Day': [day_name] * n_samples,
            'Temperature (°C)': np.random.uniform(temp_range[0], temp_range[1], n_samples),
            'Humidity (%)': np.random.uniform(humidity_range[0], humidity_range[1], n_samples),
            'Wind Speed (km/h)': np.random.uniform(wind_range[0], wind_range[1], n_samples),
            'Precipitation (mm)': np.random.uniform(precip_range[0], precip_range[1], n_samples),
        })
        # Add a target variable (e.g., temperature or weather condition)
        day_data['Future Temperature (°C)'] = day_data['Temperature (°C)'] + np.random.normal(0, 3, n_samples)
        
        data_list.append(day_data)  # Add each day's data to the list
    
    # Combine all days into one DataFrame
    all_days = pd.concat(data_list, ignore_index=True)
    return all_days

# EDA
def perform_eda(data):
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Temperature (°C)'], bins=20, kde=True, ax=ax)
        ax.set_title("Temperature Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Humidity vs. Temperature")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data['Humidity (%)'], y=data['Temperature (°C)'], hue=data['Day'], ax=ax)
        ax.set_title("Humidity vs. Temperature")
        st.pyplot(fig)

# Preprocessing
def preprocess_data(data):
    data = pd.get_dummies(data, columns=['Day'], drop_first=True)  # Encode day
    return data

# Model Training
def train_model(X_train, y_train, model_choice):
    if model_choice == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Display Metrics
def display_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.write(f"**{model_name} Metrics**")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

# Main App
st.title("🌦️ Weather Data Generator & Prediction")

# Sidebar Inputs
st.sidebar.header("Synthetic Data Parameters")
n_samples = st.sidebar.number_input("Number of Samples", min_value=100, value=1000)
temp_range = st.sidebar.slider("Temperature Range (°C)", min_value=-30, max_value=50, value=(-10, 35))
humidity_range = st.sidebar.slider("Humidity Range (%)", min_value=0, max_value=100, value=(20, 80))
wind_range = st.sidebar.slider("Wind Speed Range (km/h)", min_value=0, max_value=150, value=(0, 50))
precip_range = st.sidebar.slider("Precipitation Range (mm)", min_value=0, max_value=500, value=(0, 100))
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting"])

# Generate Data
data = generate_weather_data(n_samples, temp_range, humidity_range, wind_range, precip_range)
st.write("Generated Data:")
st.write(data.head())

# Perform EDA
perform_eda(data)

# Preprocess Data
data = preprocess_data(data)
X = data.drop(columns=['Future Temperature (°C)'])
y = data['Future Temperature (°C)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = train_model(X_train, y_train, model_choice)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Model Evaluation")
display_metrics(y_test, y_pred, model_choice)

# Scatter Plot: Actual vs Predicted
st.subheader("Actual vs Predicted Future Temperature")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, label="Predicted", color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Prediction")
ax.set_title("Actual vs Predicted")
ax.set_xlabel("Actual Temperature (°C)")
ax.set_ylabel("Predicted Temperature (°C)")
ax.legend()
st.pyplot(fig)
