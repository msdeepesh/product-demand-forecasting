import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"

# Set page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting & Inventory Optimizer",
    page_icon="📦",
    layout="wide",
)

# Load the trained model and features
@st.cache_resource
def load_assets():
    model = joblib.load(ARTIFACT_DIR / "demand_rf_model.pkl")
    model_features = joblib.load(ARTIFACT_DIR / "model_features.pkl")
  return model, model_features


model, model_features = load_assets()

st.title("📦 Retail & Supply Chain Demand Forecasting Dashboard")
st.markdown(
    "Predict product demand using machine learning and optimize inventory"
    " levels to prevent stockouts[cite: 1]."
)

# Sidebar for user inputs
st.sidebar.header("Input Parameters")


def user_input_features():
  store_id = st.sidebar.selectbox(
      "Store ID", ["S001", "S002", "S003", "S004", "S005"]
  )
  product_id = st.sidebar.selectbox("Product ID", ["P0001", "P0002", "P0003"])
  category = st.sidebar.selectbox(
      "Category", ["Electronics", "Clothing", "Home", "Groceries"]
  )
  region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])

  price = st.sidebar.slider("Product Price ($)", 10.0, 200.0, 75.0)
  discount = st.sidebar.slider("Discount (%)", 0, 50, 10)
  competitor_pricing = st.sidebar.slider(
      "Competitor Pricing ($)", 10.0, 200.0, 80.0
  )
  inventory_level = st.sidebar.number_input(
      "Current Inventory Level", min_value=0, max_value=1000, value=150
  )
  units_sold = st.sidebar.number_input(
      "Recent Units Sold (Past Period)", min_value=0, max_value=500, value=100
  )

  weather = st.sidebar.selectbox(
      "Weather Condition", ["Clear", "Rainy", "Snowy", "Cloudy"]
  )
  promotion = st.sidebar.selectbox(
      "Active Promotion", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
  )
  seasonality = st.sidebar.selectbox(
      "Seasonality", ["Spring", "Summer", "Autumn", "Winter"]
  )
  epidemic = st.sidebar.selectbox(
      "Epidemic Factor", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
  )

  date = st.sidebar.date_input("Prediction Date")

  data = {
      "Store ID": store_id,
      "Product ID": product_id,
      "Category": category,
      "Region": region,
      "Inventory Level": inventory_level,
      "Units Sold": units_sold,
      "Units Ordered": units_sold * 1.5,  # Estimated baseline
      "Price": price,
      "Discount": discount,
      "Weather Condition": weather,
      "Promotion": promotion,
      "Competitor Pricing": competitor_pricing,
      "Seasonality": seasonality,
      "Epidemic": epidemic,
      "Date": pd.to_datetime(date),
  }
  return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# Display current selections
st.subheader("Current Parameter Selection")
st.dataframe(input_df.drop(columns=["Date"]))

# Preprocess input data to match model training features
input_df["Year"] = input_df["Date"].dt.year
input_df["Month"] = input_df["Date"].dt.month
input_df["Day"] = input_df["Date"].dt.day
input_df["DayOfWeek"] = input_df["Date"].dt.dayofweek

categorical_cols = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align columns with training data features (fill missing dummy columns with 0)
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# Prediction Button
if st.button("Predict Demand & Optimize Inventory"):
  prediction = model.predict(input_encoded)[0]
  predicted_demand = round(prediction)

  current_inventory = input_df["Inventory Level"].values[0]

  st.success(f"### Predicted Demand Volume: **{predicted_demand} units**")

  # Inventory Optimization Logic
  col1, col2 = st.columns(2)
  with col1:
    st.metric(label="Current Inventory", value=current_inventory)
  with col2:
    stock_diff = predicted_demand - current_inventory
    if stock_diff > 0:
      st.metric(
          label="Recommended Action",
          value=f"Order {stock_diff} more units",
          delta="Risk of Stockout",
          delta_color="inverse",
      )
    else:
      st.metric(
          label="Recommended Action",
          value="Sufficient Inventory",
          delta="Optimal Stock Level",
          delta_color="normal",
      )