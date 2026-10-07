from pathlib import Path
import pandas as pd
#from utils import printm, print_info
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import importlib

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

##########################################
# EDA (Exploratory Data Analysis) PIPELINE
##########################################
import eda_pipeline
importlib.reload(eda_pipeline)
from eda_pipeline import EDAPipeline
eda_pipe = EDAPipeline()
df = eda_pipe.run(df)

##################################
# PREPROCESSING PIPELINE
##################################
import preprocessing
importlib.reload(preprocessing)
from preprocessing import DataPreprocessingPipeline
pipeline = DataPreprocessingPipeline()
df = pipeline.fit_transform(df)

##################################
# FEATURE ENGINEERING PIPELINE
##################################

import feature_engineering
importlib.reload(feature_engineering)
from feature_engineering import FeatureEngineeringPipeline

feature_pipeline = FeatureEngineeringPipeline()
df = feature_pipeline.fit_transform(df)

processed_data = df
split_date = '2024-01-01'
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

print(f'Training Set Shape: {train_df.shape}')
print(f'Testing Set Shape: {test_df.shape}')

