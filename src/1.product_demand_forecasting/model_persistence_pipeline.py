import joblib


class ModelPersistencePipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def run(self, model, feature_columns):
        self.printm("\n7)Saving model artifacts...")
        joblib.dump(model, "demand_rf_model.pkl")
        joblib.dump(feature_columns, "model_features.pkl")
        self.printm("\n7.1)Model saved successfully as 'demand_rf_model.pkl'!")