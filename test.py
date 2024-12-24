import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

def generate_weather_data(n_samples, temp_range, humidity_range, wind_speed_range, rainfall_range):
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Repeat the days in sequence to match the number of samples
    day = np.tile(days, n_samples // len(days))  # Repeat the sequence
    if len(day) < n_samples:
        day = np.concatenate([day, days[:n_samples - len(day)]])  # Ensure we match the exact number of samples
    
    # Debugging: Display the first few days to ensure proper order
    print("Days sequence (first few entries):", day[:10])  # Output first 10 for inspection
    
    # Assign random values to the weather parameters
    temperature = np.random.uniform(temp_range[0], temp_range[1], n_samples)
    humidity = np.random.uniform(humidity_range[0], humidity_range[1], n_samples)
    wind_speed = np.random.uniform(wind_speed_range[0], wind_speed_range[1], n_samples)
    rainfall = np.random.uniform(rainfall_range[0], rainfall_range[1], n_samples)
    weather_condition = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], n_samples, p=[0.6, 0.3, 0.1])
    
    # Create DataFrame with weather data
    data = pd.DataFrame({
        'Day': day,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Wind Speed (km/h)': wind_speed,
        'Rainfall (mm)': rainfall,
        'Weather Condition': weather_condition
    })
    
    return data

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Rainfall (mm)']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Rainfall (mm)'])
    data[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Rainfall (mm)']] = scaled_df
    st.write(data.head())
    return data

def display_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    st.write(f"**{model_name} Evaluation Metrics**")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    return mse, r2, mae

def perform_eda(data):
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Temperature (°C)'], bins=20, kde=True, ax=ax)
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Rainfall vs Temperature")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data['Temperature (°C)'], y=data['Rainfall (mm)'], hue=data['Weather Condition'], palette="viridis", ax=ax)
        ax.set_title("Rainfall vs Temperature")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Rainfall (mm)")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

st.title("Weather Data Generator and Predictor")

# Sidebar inputs for synthetic data generation
st.sidebar.header("Input Parameters")
n_samples = st.sidebar.number_input("Number of Samples", min_value=100, value=1000)
temp_min, temp_max = st.sidebar.slider("Temperature Range (°C)", min_value=-30, max_value=50, value=(-10, 35))
humidity_min, humidity_max = st.sidebar.slider("Humidity Range (%)", min_value=0, max_value=100, value=(10, 90))
wind_min, wind_max = st.sidebar.slider("Wind Speed Range (km/h)", min_value=0, max_value=150, value=(0, 50))
rain_min, rain_max = st.sidebar.slider("Rainfall Range (mm)", min_value=0, max_value=500, value=(0, 100))

# Generate synthetic weather data
data = generate_weather_data(n_samples, (temp_min, temp_max), (humidity_min, humidity_max), (wind_min, wind_max), (rain_min, rain_max))
st.write("Generated Weather Data:")
st.write(data.head())

# Perform EDA
perform_eda(data)

st.header("Model Training and Evaluation")

# Preprocess the data
data = pd.get_dummies(data, columns=['Weather Condition'], drop_first=True)
data = preprocess_data(data)

# Train-test split
X = data.drop(columns=['Rainfall (mm)'])
y = data['Rainfall (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
st.subheader("Random Forest Regressor")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_mse, rf_r2, rf_mae = display_metrics(y_test, rf_y_pred, "Random Forest")

# Linear Regression
st.subheader("Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_mse, lr_r2, lr_mae = display_metrics(y_test, lr_y_pred, "Linear Regression")

# Scatter Plot: Actual vs Predicted Rainfall
st.subheader("Actual vs Predicted Rainfall")
fig, ax = plt.subplots()
ax.scatter(y_test, rf_y_pred, alpha=0.6, label="Random Forest", color='blue')
ax.scatter(y_test, lr_y_pred, alpha=0.6, label="Linear Regression", color='orange')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
ax.set_title("Actual vs Predicted Rainfall")
ax.set_xlabel("Actual Rainfall (mm)")
ax.set_ylabel("Predicted Rainfall (mm)")
ax.legend()
st.pyplot(fig)
