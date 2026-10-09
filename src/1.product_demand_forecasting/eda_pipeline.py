import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDAPipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def dataset_overview(self, df):
        self.printm(f'\n2.1)Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
        self.printm(f'\n2.2)Dataset info:')
        self.printm(f'{df.info()}')
        self.printm(f'\n2.3)Dataset description:')
        self.printm(f'{df.describe(include='all').T}')
        self.printm(f'\n2.4)Missing values count:{df.isnull().sum()}')
        self.printm(f'\n2.5)Duplicate rows: {df.duplicated().sum()}')
        self.printm(f'\n2.6)Unique values count:\n{df.nunique()}')
        self.printm(f'\n2.7)Dataset head:\n{df.head()}')
        self.printm(f'\n2.8)Dataset tail:\n{df.tail()}')
        return df

    def categorical_distributions(self, df):
        self.printm(f'\n2.9)Categorical Distributions:')
        self.printm('2.9.1. STORE ID DISTRIBUTIONS')
        self.printm(df['Store ID'].value_counts())

        self.printm('\n2.9.2. PRODUCT CATEGORIES DISTRIBUTIONS')
        cat_counts = df['Category'].value_counts()
        cat_pcts = df['Category'].value_counts(normalize=True) * 100
        for cat in cat_counts.index:
            self.printm(f'- {cat}: {cat_counts[cat]} rows ({cat_pcts[cat]:.1f}%)')

        self.printm('\n2.9.3. REGION DISTRIBUTIONS')
        self.printm(df['Region'].value_counts())

        self.printm('\n2.9.4. WEATHER CONDITION DISTRIBUTIONS')
        weather_counts = df['Weather Condition'].value_counts()
        weather_pcts = df['Weather Condition'].value_counts(normalize=True) * 100
        for w in weather_counts.index:
            self.printm(f'- {w}: {weather_counts[w]} rows (~{weather_pcts[w]:.2f}%)')

        self.printm('\n2.9.5. EPIDEMIC DISTRIBUTIONS')
        epidemic_counts = df['Epidemic'].value_counts()
        epidemic_pcts = df['Epidemic'].value_counts(normalize=True) * 100
        for ep in epidemic_counts.index:
            self.printm(f'- Status {ep}: {epidemic_counts[ep]} rows ({epidemic_pcts[ep]:.1f}%)')
        return df

    def minimal_distribution_charts(self, df):
        self.printm(f'\n2.10) Distribution Charts:')
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
        plt.savefig("eda_minimal_distribution_charts.png")
        self.printm("\n2.10.1) Distribution charts saved successfully as 'eda_minimal_distribution_charts.png'!")
        return df
    
    def visualization_charts(self, df):
        # 4. Visualizations
        self.printm(f'\n2.11) Visualization Charts:')
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Distribution of Demand
        sns.histplot(df["Demand"], kde=True, ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title("Distribution of Product Demand")
        axes[0, 0].set_xlabel("Demand (Units)")

        # Plot 2: Demand across Categories
        sns.barplot(
            x="Category",
            y="Demand",
            data=df,
            ax=axes[0, 1],
            palette="Set2",
            errorbar=None,
        )
        axes[0, 1].set_title("Average Demand by Product Category")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Impact of Promotion on Demand
        sns.boxplot(x="Promotion", y="Demand", data=df, ax=axes[1, 0], palette="Set3")
        axes[1, 0].set_title("Demand vs. Promotion Status (0 = No, 1 = Yes)")

        # Plot 4: Correlation Heatmap of Numerical Features
        numeric_cols = [
            "Inventory Level",
            "Units Sold",
            "Price",
            "Discount",
            "Competitor Pricing",
            "Demand",
        ]
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
        axes[1, 1].set_title("Correlation Matrix")

        plt.tight_layout()
        plt.savefig("eda_visualizations.png")
        print("\n2.11.1)Visualization charts saved successfully as 'eda_visualizations.png'!")

    def run(self, df):
        self.printm('\n2)Starting EDA pipeline...')
        df = self.dataset_overview(df)
        df = self.categorical_distributions(df)
        self.minimal_distribution_charts(df)
        self.visualization_charts(df)
        self.printm('\n2.12) EDA pipeline completed successfully!')
        return df
