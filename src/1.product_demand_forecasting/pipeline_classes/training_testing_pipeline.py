from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class TrainingTestingPipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def run(self, df):
        self.printm("\n5)Starting training and testing pipeline...")
        X = df.drop(columns=["Date", "Demand"])
        y = df["Demand"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.printm(
            f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}"
        )
        self.printm("\n5.1)Training Random Forest Regressor...")

        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=20
        )
        rf_model.fit(X_train, y_train)
        self.printm("\n5.2)Training and testing pipeline completed successfully!")
        return rf_model, X_train, X_test, y_train, y_test