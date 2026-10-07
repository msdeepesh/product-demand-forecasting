import pandas as pd
import matplotlib.pyplot as plt


class EDAPipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def dataset_overview(self, df):
        self.printm(f'\nDataset Shape:\n{df.shape[0]} rows, {df.shape[1]} columns')
        self.printm(f'\nDataset info:\n')
        self.printm(f'{df.info()}')
        self.printm(f'\nDataset description:\n')
        self.printm(f'{df.describe(include='all').T}')
        self.printm(f'\nMissing values count:\n{df.isnull().sum()}')
        self.printm(f'\nDuplicate rows: {df.duplicated().sum()}')
        self.printm(f'\nUnique values count:\n{df.nunique()}')
        self.printm(f'\nDataset head:\n{df.head()}')
        self.printm(f'\nDataset tail:\n{df.tail()}')
        return df

    def categorical_distributions(self, df):
        self.printm('=== 1. STORE ID DISTRIBUTIONS ===')
        self.printm(df['Store ID'].value_counts())

        self.printm('\n=== 2. PRODUCT CATEGORIES DISTRIBUTIONS ===')
        cat_counts = df['Category'].value_counts()
        cat_pcts = df['Category'].value_counts(normalize=True) * 100
        for cat in cat_counts.index:
            self.printm(f'- {cat}: {cat_counts[cat]} rows ({cat_pcts[cat]:.1f}%)')

        self.printm('\n=== 3. REGION DISTRIBUTIONS ===')
        self.printm(df['Region'].value_counts())

        self.printm('\n=== 4. WEATHER CONDITION DISTRIBUTIONS ===')
        weather_counts = df['Weather Condition'].value_counts()
        weather_pcts = df['Weather Condition'].value_counts(normalize=True) * 100
        for w in weather_counts.index:
            self.printm(f'- {w}: {weather_counts[w]} rows (~{weather_pcts[w]:.2f}%)')

        self.printm('\n=== 5. EPIDEMIC DISTRIBUTIONS ===')
        epidemic_counts = df['Epidemic'].value_counts()
        epidemic_pcts = df['Epidemic'].value_counts(normalize=True) * 100
        for ep in epidemic_counts.index:
            self.printm(f'- Status {ep}: {epidemic_counts[ep]} rows ({epidemic_pcts[ep]:.1f}%)')
        return df

    def minimal_distribution_charts(self, df):
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
        return df

    def run(self, df):
        self.printm('Starting EDA pipeline...')
        df = self.dataset_overview(df)
        df = self.categorical_distributions(df)
        self.minimal_distribution_charts(df)
        self.printm('EDA pipeline completed successfully!')
        return df
