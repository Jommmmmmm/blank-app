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

def preprocess_data(data):
    data = pd.get_dummies(data, columns=['Day'], drop_first=True)
    data = data.drop(columns=[col for col in data.columns if 'Day_' in col])  # Drop 'Day' columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)'])
    data[['Discount (%)', 'Marketing Spend ($)', 'Competitor Price ($)']] = scaled_df
    st.write(data.head())
    return data

def generate_synthetic_data(n_samples, avg_marketing_spend, std_marketing_spend, min_discount, max_discount, competitor_price_min, competitor_price_max):
    np.random.seed(42)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day = np.random.choice(days, n_samples)
    discount = np.random.uniform(min_discount, max_discount, n_samples)
    marketing_spend = np.random.normal(avg_marketing_spend, std_marketing_spend, n_samples)
    competitor_price = np.random.uniform(competitor_price_min, competitor_price_max, n_samples)
    base_sales = 50 + marketing_spend / 1000 - discount / 2
    sales_count = np.random.poisson(base_sales).clip(min=0)
    return pd.DataFrame({
        'Day': day,
        'Discount (%)': discount,
        'Marketing Spend ($)': marketing_spend,
        'Competitor Price ($)': competitor_price,
        'Sales Count': sales_count
    })

# Evaluation Metrics
def display_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    st.write(f"**{model_name} Evaluation Metrics**")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    st.write(f"""
    **Implications for {model_name}:**
    - **MSE**: Lower values indicate better performance.
    - **R²**: Higher values (close to 1) indicate the model explains more variance in the data.
    - **MAE**: Lower values indicate less average error between actual and predicted values.
    """)
    return mse, r2, mae

def perform_eda(data):
    colu1, colu2 = st.columns(2, gap='medium', border=True)
    with colu1:
      # Sales Count Distribution
      st.subheader("Sales Count Distribution")
      fig, ax = plt.subplots()
      sns.histplot(data['Sales Count'], bins=20, kde=True, ax=ax)
      ax.set_title("Sales Count Distribution")
      ax.set_xlabel("Sales Count")
      ax.set_ylabel("Frequency")
      st.pyplot(fig)
    with colu1:
      st.write("""
      **Implications**:
      - **X-axis (`Sales Count`)**: Represents the number of items sold in a given scenario.
      - **Y-axis (`Frequency`)**: Shows how often a particular sales count appears in the dataset.
      - The histogram, combined with the KDE curve, indicates that the distribution of Sales Count is approximately normal, with the highest frequency around a sales count of 55. This suggests that most sales fall within this range, indicating a central tendency. 
      - **Peaks in the KDE Curve:** Represent the most common sales counts. Flat regions on either side of the peak indicate fewer occurrences of very low or very high sales counts.
      """)

      with colu2:
         # Scatter Plot: Sales Count vs Marketing Spend
         st.subheader("Sales Count vs Marketing Spend")
         fig, ax = plt.subplots()
         sns.scatterplot(x=data['Marketing Spend ($)'], y=data['Sales Count'], hue=data['Day'], palette="viridis", ax=ax)
         ax.set_title("Sales Count vs Marketing Spend")
         ax.set_xlabel("Marketing Spend ($)")
         ax.set_ylabel("Sales Count")
         st.pyplot(fig)
      with colu2:
         st.write("""
         **Implications**:
         - **X-axis (`Marketing Spend ($)`)**: Represents the amount spent on marketing in dollars, ranging from 2000 to $9000.
         - **Y-axis (`Sales Count`)**: Represents the number of items sold.
         - **Legend (`Day`)**: Indicates the day of the week corresponding to the data points. Each color represents a specific day.
         - The scatter plot shows the relationship between `Marketing Spend` and `Sales Count`. A visible positive trend suggests that higher marketing spend tends to result in more sales.
         - The color-coded points allow us to observe if certain days have stronger relationships between marketing spend and sales. For example, weekends might show higher sales with lower marketing spend.
         """)
    with st.container(border=True):
      map1, map2 = st.columns(2)
      # Correlation Heatmap
      with map1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
      with map2:
            st.write("""
            **Implications**:
            - **X-axis & Y-axis**: Represent features of the dataset (e.g., `Marketing Spend`, `Discount (%)`, etc.).
            - **Cell Values**: Show the correlation coefficient between two features.
               - Values range from -1 to 1:
               - `1`: Perfect positive correlation (as one feature increases, the other increases).
               - `0`: No correlation.
               - `-1`: Perfect negative correlation (as one feature increases, the other decreases).
               - Highly correlated features (e.g., `Marketing Spend` and `Sales Count`) suggest strong relationships, which could guide business decisions. For instance, focusing on `Marketing Spend` could optimize sales outcomes.
               """)

# Streamlit App Setup
st.title("Synthetic Data Generation and Prediction App")

# Sidebar inputs for synthetic data generation
st.sidebar.header("Input Parameters")
n_samples = st.sidebar.number_input("Number of Samples", min_value=100, value=10000)
avg_marketing_spend = st.sidebar.number_input("Average Marketing Spend ($)", value=10000)
std_marketing_spend = st.sidebar.number_input("Standard Deviation of Marketing Spend ($)", value=10000)
min_discount = st.sidebar.slider("Minimum Discount (%)", min_value=0, max_value=30, value=10)
max_discount = st.sidebar.slider("Maximum Discount (%)", min_value=10, max_value=50, value=20)
competitor_price_min = st.sidebar.number_input("Competitor Price Min ($)", value=1000)
competitor_price_max = st.sidebar.number_input("Competitor Price Max ($)", value=5000)

# Generate synthetic data
data = generate_synthetic_data(n_samples, avg_marketing_spend, std_marketing_spend, min_discount, max_discount, competitor_price_min, competitor_price_max)
st.write("Generated Data:")
st.write(data.head())
# Description of each feature
st.markdown("""
### Feature Descriptions:

1. **`Day`** (Categorical):
   - The day of the week (Monday, Tuesday, Wednesday, etc.) on which sales occurred. This feature helps to see if sales are higher on weekends or weekdays.
   
2. **`Discount (%)`** (Numerical):
   - The percentage discount applied to the product. It is a random value between the minimum and maximum discount values. Higher discounts generally lead to higher sales.

3. **`Marketing Spend ($)`** (Numerical):
   - The amount spent on marketing activities on a given day. This feature is drawn from a normal distribution with the specified mean and standard deviation. More marketing spending usually increases sales.

4. **`Competitor Price ($)`** (Numerical):
   - The price of the product set by the competitor. It is randomly generated within a user-defined price range. Competitor pricing can impact sales, especially if your price is higher than the competitor's.

5. **`Sales Count`** (Target Variable - Numerical):
   - The number of units sold on a particular day. This is the target variable that we're trying to predict. It is calculated based on marketing spend and discount, and adjusted with random noise using a Poisson distribution.
""")

# Perform EDA
st.header("Exploratory Data Analysis")
perform_eda(data)

st.header("Model Training and Evaluation")
# Train the model
with st.container(border=True):
   cols1, cols2 = st.columns(2)
   # Preprocess the data
   data = preprocess_data(data)
   # Train-test split
   X = data.drop(columns=['Sales Count'])
   y = data['Sales Count']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Random Forest Regressor
   st.subheader("Random Forest Regressor")
   rf_model = RandomForestRegressor(random_state=42)
   rf_model.fit(X_train, y_train)
   rf_y_pred = rf_model.predict(X_test)
   with cols2:
      # Display metrics for Random Forest
      st.write("### Random Forest Metrics")
      rf_mse, rf_r2, rf_mae = display_metrics(y_test, rf_y_pred, "Random Forest")
      # Feature Importance for Random Forest
   with cols1:
      st.subheader("Random Forest Feature Importance")
      rf_importances = rf_model.feature_importances_
      feature_names = X.columns
      fig, ax = plt.subplots()
      ax.barh(feature_names, rf_importances, color="skyblue")
      ax.set_title("Feature Importance in Random Forest")
      ax.set_xlabel("Importance")
      ax.set_ylabel("Feature")
      st.pyplot(fig)

      # Linear Regression
      st.subheader("Linear Regression")
      lr_model = LinearRegression()
      lr_model.fit(X_train, y_train)
      lr_y_pred = lr_model.predict(X_test)

      # Display metrics for Linear Regression
      st.write("### Linear Regression Metrics")
      lr_mse, lr_r2, lr_mae = display_metrics(y_test, lr_y_pred, "Linear Regression")


# Scatter Plot: Actual vs Predicted Sales (Both Models)
st.subheader("Actual vs Predicted Sales: Random Forest vs Linear Regression")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, rf_y_pred, alpha=0.6, label="Random Forest", color='blue')
ax.scatter(y_test, lr_y_pred, alpha=0.6, label="Linear Regression", color='orange')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
ax.set_title("Actual vs Predicted Sales")
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.legend()
st.pyplot(fig)

plt.figure(figsize=(10, 6)) 
sns.regplot(x='Marketing Spend ($)', y='Sales Count', data=data, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}) 
plt.title('Sales Count vs Marketing Spend with Regression Line') 
plt.xlabel('Marketing Spend ($)') 
plt.ylabel('Sales Count') 
plt.show()


