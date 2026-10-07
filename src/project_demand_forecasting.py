from pathlib import Path
import pandas as pd
#from utils import printm, print_info
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

##################################
# Loading dataset
##################################
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / 'data' / 'demand_forecasting.csv'
try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {csv_path} was not found. Please update the path.")

##################################
# EDA (Exploratory Data Analysis)
##################################

# import importlib
# import eda_pipeline
# importlib.reload(eda_pipeline)
# from eda_pipeline import EDAPipeline

# # Initialize and execute the modular EDA pipeline
# eda_pipe = EDAPipeline()
# eda_pipe.run(df)

# Number of columns and rows in the dataset
print(f'\nDataset Shape:\n{df.shape[0]} rows, {df.shape[1]} columns')

# Dataset information
print(f'\nDataset info:\n')
print(f'{df.info()}')

# Dataset description
print(f'\nDataset description:\n')
print(f'{df.describe()}')

# Missing values count
print(f'\nMissing values count:\n{df.isnull().sum()}')

# Duplicate rows
print(f'\nDuplicate rows: {df.duplicated().sum()}')

# Unique values count
print(f'\nUnique values count:\n{df.nunique()}')

# First few rows of the dataset
print(f'\nDataset head:\n{df.head()}')

# Last few rows of the dataset
print(f'\nDataset tail:\n{df.tail()}')

# Categorical Distribution Analysis
print('=== 1. STORE ID DISTRIBUTIONS ===')
print(df['Store ID'].value_counts())

print('\n=== 2. PRODUCT CATEGORIES DISTRIBUTIONS ===')
cat_counts = df['Category'].value_counts()
cat_pcts = df['Category'].value_counts(normalize=True) * 100
for cat in cat_counts.index:
  print(f'- {cat}: {cat_counts[cat]} rows ({cat_pcts[cat]:.1f}%)')

print('\n=== 3. REGION DISTRIBUTIONS ===')
print(df['Region'].value_counts())

print('\n=== 4. WEATHER CONDITION DISTRIBUTIONS ===')
weather_counts = df['Weather Condition'].value_counts()
weather_pcts = df['Weather Condition'].value_counts(normalize=True) * 100
for w in weather_counts.index:
  print(f'- {w}: {weather_counts[w]} rows (~{weather_pcts[w]:.2f}%)')

print('\n=== 5. EPIDEMIC DISTRIBUTIONS ===')
epidemic_counts = df['Epidemic'].value_counts()
epidemic_pcts = df['Epidemic'].value_counts(normalize=True) * 100
for ep in epidemic_counts.index:
  print(f'- Status {ep}: {epidemic_counts[ep]} rows ({epidemic_pcts[ep]:.1f}%)')

# Minimal distribution charts for categorical fields
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

plots = [
    (df['Store ID'].value_counts(), 'Store ID Distribution', 'Store ID', 'Count'),
    (df['Category'].value_counts(), 'Category Distribution', 'Category', 'Count'),
    (df['Region'].value_counts(), 'Region Distribution', 'Region', 'Count'),
    (df['Weather Condition'].value_counts(), 'Weather Condition Distribution', 'Weather', 'Count'),
    (df['Epidemic'].value_counts(), 'Epidemic Distribution', 'Epidemic', 'Count'),
]

for ax, (series, title, xlabel, ylabel) in zip(axes, plots):
    series.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)

if len(plots) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

import importlib
import preprocessing
importlib.reload(preprocessing)
from preprocessing import DataPreprocessingPipeline

##################################
# RUN PREPROCESSING PIPELINE
##################################
pipeline = DataPreprocessingPipeline()
df = pipeline.fit_transform(df)

##################################
# ADVANCED FEATURE ENGINEERING')
##################################

# A. Temporal & Cyclical Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# Cyclical Encoding (Fixed missing '*' and markdown bullets)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# B. Fourier Terms for Multi-Year Seasonality (Fixed missing subtraction '-')
t = (df['Date'] - df['Date'].min()).dt.days
df['Fourier_Sin_1'] = np.sin(2 * np.pi * t / 365.25)
df['Fourier_Cos_1'] = np.cos(2 * np.pi * t / 365.25)

# C. Grouped Lag & Rolling Window Features (Per Store & Product)
grouped = df.groupby(['Store ID', 'Product ID'])

# Lag Features (Past Demand)
df['Demand_Lag_7'] = grouped['Demand'].shift(7)
df['Demand_Lag_30'] = grouped['Demand'].shift(30)
df['Demand_Lag_365'] = grouped['Demand'].shift(365)  # Multi-year YoY lag

# Rolling Window Statistics
df['Demand_Rolling_Mean_7'] = grouped['Demand'].shift(1).rolling(window=7, min_periods=1).mean()
df['Demand_Rolling_Std_7'] = grouped['Demand'].shift(1).rolling(window=7, min_periods=1).std()

# D. Competitive Pricing Shift Features (Fixed missing subtraction '-')
df['Price_Diff'] = df['Price'] - df['Competitor Pricing']
df['Price_Ratio'] = df['Price'] / (df['Competitor Pricing'] + 1e-5)

# Handle NaN values generated from shifts/rolling windows
df = df.bfill().fillna(0)

##################################
# CATEGORICAL ENCODING & SCALING'
##################################

# One-Hot Encoding for nominal features
nominal_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Scale numerical continuous variables
scale_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Competitor Pricing', 'Price_Diff', 'Price_Ratio']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

##################################
# CHRONOLOGICAL TRAIN-TEST SPLIT
##################################

# Strict time-based split to prevent data leakage
split_date = '2024-01-01'
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

processed_data = df
train_data = train_df
test_data = test_df

print(f'Training Set Shape: {train_df.shape}')
print(f'Testing Set Shape: {test_df.shape}')

