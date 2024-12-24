import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Weather Data Generator & Prediction",
    page_icon="🌦️",
    layout="wide"
)

def generate_synthetic_data(
    n_samples, 
    avg_marketing_spend, 
    std_marketing_spend, 
    min_discount, 
    max_discount, 
    competitor_price_min, 
    competitor_price_max
):
    """
    Generate synthetic weather data ensuring all days (Monday to Sunday) are included.
    """
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Generate data such that all days are covered
    day = np.tile(days, n_samples // len(days))  # Repeat days to ensure all are included
    remaining_samples = n_samples % len(days)
    day = np.concatenate([day, np.random.choice(days, remaining_samples, replace=False)])
    
    discount = np.random.uniform(min_discount, max_discount, n_samples)
    marketing_spend = np.random.normal(avg_marketing_spend, std_marketing_spend, n_samples)
    competitor_price = np.random.uniform(competitor_price_min, competitor_price_max, n_samples)
    
    # Base sales calculation (you can modify this for weather-specific logic)
    base_sales = 50 + marketing_spend / 1000 - discount / 2
    sales_count = np.random.poisson(base_sales).clip(min=0)
    
    # Create the DataFrame
    return pd.DataFrame({
        'Day': day,
        'Discount (%)': discount,
        'Marketing Spend ($)': marketing_spend,
        'Competitor Price ($)': competitor_price,
        'Sales Count': sales_count
    })


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
