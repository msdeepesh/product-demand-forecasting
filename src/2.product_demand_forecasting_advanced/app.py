import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from advanced_model import prep, compare, sarimax, future_tree, plan


st.set_page_config(page_title="Advanced Demand Planning", layout="wide")
st.title("Advanced Demand Forecasting & Production Planning")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is None:
    st.info(
        "Required: date, sku, sales, inventory. Optional: price, promo, marketing, market, lead_time, incoming_supply."
    )
    st.stop()

data = prep(pd.read_csv(uploaded_file))
sku = st.selectbox("SKU", sorted(data.sku.unique()))

days = st.number_input("Validation holdout days", 7, 120, 30)
horizon = st.number_input("Forecast horizon", 7, 180, 30)
service = st.select_slider("Service level", [0.90, 0.95, 0.975, 0.99], value=0.95)
review = st.number_input("Review period", 1, 60, 7)

with st.spinner("Comparing models..."):
    results = compare(data, days)

try:
    results["SARIMAX"] = sarimax(data, sku, days)
except Exception:
    pass

table = pd.DataFrame(
    [{"Model": model_name, **values["metrics"]} for model_name, values in results.items()]
).sort_values("RMSE")

st.subheader("Model comparison")
st.dataframe(
    table.style.format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "WAPE": "{:.2%}", "sMAPE": "{:.2%}"}
    ),
    use_container_width=True,
)

choice = st.selectbox("Model used for forecast", list(table.Model))
selected_result = results[choice]

if choice == "SARIMAX":
    forecast_values = selected_result["model"].forecast(horizon)
    forecast = pd.DataFrame(
        {
            "date": forecast_values.index,
            "forecast_demand": np.maximum(0, forecast_values.values),
        }
    )
else:
    forecast = future_tree(selected_result["model"], data, sku, horizon)

validation = selected_result["validation"]
std_dev = float(np.std(validation.sales - validation.prediction))
last_row = data[data.sku == sku].sort_values("date").iloc[-1]

planning_df, safety_stock, reorder_point, recommended_production, stockout_days, stockout_units = plan(
    forecast,
    float(last_row.inventory),
    float(last_row.incoming_supply),
    int(last_row.lead_time),
    service,
    review,
    std_dev,
)

metrics = st.columns(6)
metrics[0].metric("Current inventory", f"{last_row.inventory:,.0f}")
metrics[1].metric("Forecast", f"{forecast.forecast_demand.sum():,.0f}")
metrics[2].metric("Safety stock", f"{safety_stock:,.0f}")
metrics[3].metric("Reorder point", f"{reorder_point:,.0f}")
metrics[4].metric("Recommended production", f"{recommended_production:,.0f}")
metrics[5].metric("Stockout days", stockout_days)

historical_data = data[data.sku == sku].tail(180)
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=historical_data.date, y=historical_data.sales, name="Historical"))
forecast_fig.add_trace(go.Scatter(x=forecast.date, y=forecast.forecast_demand, name="Forecast"))
st.plotly_chart(forecast_fig, use_container_width=True)

inventory_fig = go.Figure()
inventory_fig.add_trace(
    go.Scatter(x=planning_df.date, y=planning_df.projected_inventory, name="Projected inventory")
)
inventory_fig.add_hline(y=0, line_dash="dash", annotation_text="Stockout")
inventory_fig.add_hline(y=safety_stock, line_dash="dot", annotation_text="Safety stock")
inventory_fig.add_hline(y=reorder_point, line_dash="dot", annotation_text="Reorder point")
st.plotly_chart(inventory_fig, use_container_width=True)

st.metric("Projected stockout units", f"{stockout_units:,.0f}")
st.dataframe(planning_df, use_container_width=True)
st.download_button(
    "Download planning CSV",
    planning_df.to_csv(index=False).encode(),
    "planning_output.csv",
    "text/csv",
)

st.markdown(
    "### Methods: HistGradientBoosting + RandomForest + SARIMAX. The best model can be selected using validation RMSE. The selected forecast feeds safety stock, reorder point, stockout detection and production recommendation."
)
