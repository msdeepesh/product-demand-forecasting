import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluationPipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def run(self, model, X_test, y_test):
        self.printm("\n6.Evaluating model performance...")
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        self.printm("\n6.1)--- Random Forest Performance ---")
        self.printm(f"\n6.1.1)Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        self.printm(f"\n6.1.2)Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        self.printm(f"\n6.1.3)R-Squared (R2 Score): {metrics['r2']:.4f}")
        self.printm("\n6.2)Model evaluation completed successfully!")
        return metrics