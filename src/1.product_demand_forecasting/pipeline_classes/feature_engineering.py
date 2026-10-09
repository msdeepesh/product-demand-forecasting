import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineeringPipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def add_temporal_features(self, df):
        self.printm('\n4.1)Adding temporal features...')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        return df

    def add_cyclical_features(self, df):
        self.printm('\n4.2)Adding cyclical time features...')
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        return df

    def add_fourier_features(self, df):
        self.printm('\n4.3)Adding Fourier seasonality features...')
        t = (df['Date'] - df['Date'].min()).dt.days
        df['Fourier_Sin_1'] = np.sin(2 * np.pi * t / 365.25)
        df['Fourier_Cos_1'] = np.cos(2 * np.pi * t / 365.25)
        return df

    def add_lag_and_rolling_features(self, df):
        self.printm('\n4.4)Adding lag and rolling demand features...')
        grouped = df.groupby(['Store ID', 'Product ID'])
        df['Demand_Lag_7'] = grouped['Demand'].shift(7)
        df['Demand_Lag_30'] = grouped['Demand'].shift(30)
        df['Demand_Lag_365'] = grouped['Demand'].shift(365)

        df['Demand_Rolling_Mean_7'] = grouped['Demand'].shift(1).rolling(window=7, min_periods=1).mean()
        df['Demand_Rolling_Std_7'] = grouped['Demand'].shift(1).rolling(window=7, min_periods=1).std()
        return df

    def add_competitive_pricing_features(self, df):
        self.printm('\n4.5)Adding competitive pricing features...')
        df['Price_Diff'] = df['Price'] - df['Competitor Pricing']
        df['Price_Ratio'] = df['Price'] / (df['Competitor Pricing'] + 1e-5)
        return df

    def handle_generated_nulls(self, df):
        self.printm('\n4.6)Filling generated missing values...')
        df = df.bfill().fillna(0)
        return df

    def encode_and_scale(self, df):
        self.printm('\n4.7)Encoding categorical features and scaling numeric columns...')
        nominal_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

        scale_cols = [
            'Inventory Level',
            'Units Sold',
            'Units Ordered',
            'Price',
            'Discount',
            'Competitor Pricing',
            'Price_Diff',
            'Price_Ratio',
        ]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        return df

    def fit_transform(self, df):
        self.printm('\n4)Starting Feature Engineering Pipeline...')
        df = self.add_temporal_features(df)
        # df = self.add_cyclical_features(df)
        # df = self.add_fourier_features(df)
        df = self.add_lag_and_rolling_features(df)
        df = self.add_competitive_pricing_features(df)
        df = self.handle_generated_nulls(df)
        df = self.encode_and_scale(df)
        self.printm('\n4.8)Feature Engineering Pipeline completed successfully!')
        return df
