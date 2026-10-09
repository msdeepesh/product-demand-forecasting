import pandas as pd
import numpy as np

class DataPreprocessingPipeline:
    def __init__(self, printm_func=None):
        # Pass printm function if available, otherwise fallback to standard print
        self.printm = printm_func if printm_func is not None else print
        self.upper_bound = None
        self.lower_bound = None

    def check_missing_values(self, df):
        self.printm('\n3.1)Checking for missing values in each column...')
        missing_values = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        
        if missing_values.sum() > 0:
            raise ValueError("revisit preprocessing part: Missing values detected in the dataset!")
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_percentage
        })
        self.printm(f'\nMissing values: {str(missing_df)}')
        return df

    def handle_duplicates(self, df):
        self.printm('\n3.2)Checking for duplicate rows...')
        duplicates_count = df.duplicated().sum()
        if duplicates_count > 0:
            self.printm(f'\nFound {duplicates_count} duplicated rows in the dataset. Removing duplicates...')
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            self.printm('\nNo duplicate rows found in the dataset.')
        return df

    def parse_and_sort_dates(self, df):
        self.printm('\n3.3)Converting and sorting dates...')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
        return df

    def sanitize_categoricals(self, df):
        self.printm('\n3.4)Sanitizing categorical text fields...')
        cat_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        return df

    def validate_numeric_bounds(self, df):
        self.printm('\n3.5)Checking extreme negative/invalid errors...')
        numeric_checks = [
            'Inventory Level',
            'Units Sold',
            'Units Ordered',
            'Price',
            'Discount',
            'Demand',
        ]
        for col in numeric_checks:
            invalid_count = (df[col] < 0).sum()
            if invalid_count > 0:
                self.printm(f'Warning: Found {invalid_count} negative values in {col}. Clipping to 0.')
                df[col] = df[col].clip(lower=0)
        return df

    def manage_outliers(self, df):
        self.printm(
            "\n3.6) Managing extreme outliers."
        )
        columns_to_check = ["Demand", "Inventory Level", "Units Sold"]

        self.outlier_bounds = {}

        for col in columns_to_check:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR

            self.outlier_bounds[col] = {
                "lower": lower_bound,
                "upper": upper_bound,
            }

            outliers_count = (
                (df[col] > upper_bound) | (df[col] < lower_bound)
            ).sum()
            self.printm(
                f"\n3.6.1) Detected {outliers_count} outlier points in '{col}' based"
                " on IQR bounds."
            )

            # df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def fit_transform(self, df):
        self.printm("\n3)Starting Data Preprocessing Pipeline...")
        df_cleaned = df.copy()
        
        # Run pipeline steps sequentially
        df_cleaned = self.check_missing_values(df_cleaned)
        df_cleaned = self.handle_duplicates(df_cleaned)
        df_cleaned = self.parse_and_sort_dates(df_cleaned)
        df_cleaned = self.sanitize_categoricals(df_cleaned)
        df_cleaned = self.validate_numeric_bounds(df_cleaned)
        df_cleaned = self.manage_outliers(df_cleaned)
        
        self.printm("\n3.7)Data Preprocessing Pipeline completed successfully!")
        return df_cleaned
