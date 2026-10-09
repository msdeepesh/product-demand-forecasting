# Advanced Product Demand Forecasting & Production Planning
This version compares three forecasting approaches:
1. HistGradientBoostingRegressor — nonlinear tree-based ML
2. RandomForestRegressor — tree ensemble benchmark
3. SARIMAX — statistical time-series benchmark with weekly seasonality

The models are evaluated on a time-ordered holdout using MAE, RMSE, WAPE and sMAPE. The selected forecast feeds:
- safety stock
- reorder point
- projected inventory
- stockout days/units
- recommended production quantity

Install:
pip install -r requirements.txt
python generate_sample.py
streamlit run app.py

Required data: date, sku, sales, inventory.
Recommended additional data: price, promotion, marketing spend, market index, lead time, incoming supply, holding cost, stockout cost.

For real deployment, add planned future promotions/prices, purchase orders, production capacity, minimum batch sizes, supplier lead-time distributions and lost-sales/backorder data.
