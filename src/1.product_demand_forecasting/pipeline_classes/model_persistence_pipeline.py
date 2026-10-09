import joblib
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model_artifacts"


class ModelPersistencePipeline:
    def __init__(self, printm_func=None):
        self.printm = printm_func if printm_func is not None else print

    def run(self, model, feature_columns):
        self.printm("\n7)Saving model artifacts...")
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACT_DIR / "demand_rf_model.pkl")
        joblib.dump(feature_columns, ARTIFACT_DIR / "model_features.pkl")
        self.printm("\n7.1)Model artifacts saved successfully in 'model_artifacts/'!")