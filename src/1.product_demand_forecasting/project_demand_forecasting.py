

import importlib
import utils

######################
# Pre-Requsites Check
######################
importlib.reload(utils)
utils.confirm_setup(
    "Have you read 'Pre-requisites.txt' and installed the required packages? (yes/no): "
)

##################
# Loading dataset
##################
df = utils.load_dataset()
if df is None:
    raise SystemExit(1)

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

####################
# TRAINING PIPELINE
####################
import training_testing_pipeline
importlib.reload(training_testing_pipeline)
from training_testing_pipeline import TrainingTestingPipeline

training_pipeline = TrainingTestingPipeline()
rf_model, X_train, X_test, y_train, y_test = training_pipeline.run(df)

############################
# MODEL EVALUATION PIPELINE
############################
import model_evaluation_pipeline
importlib.reload(model_evaluation_pipeline)
from model_evaluation_pipeline import ModelEvaluationPipeline

evaluation_pipeline = ModelEvaluationPipeline()
evaluation_metrics = evaluation_pipeline.run(rf_model, X_test, y_test)

####################
# MODEL PERSISTENCE
####################
import model_persistence_pipeline
importlib.reload(model_persistence_pipeline)
from model_persistence_pipeline import ModelPersistencePipeline

persistence_pipeline = ModelPersistencePipeline()
persistence_pipeline.run(rf_model, X_train.columns.tolist())

#################
# POST EXECUTION 
#################
utils.open_file("Post execution.txt")
