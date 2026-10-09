

import importlib
import common.utils as utils

######################
# Pre-Requsites Check
######################
importlib.reload(utils)
utils.confirm_setup(
    "Have you read 'text_files/Pre-requisites.txt' and installed the required packages? (yes/no): "
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
import pipeline_classes.eda_pipeline as eda_pipeline
importlib.reload(eda_pipeline)
from pipeline_classes.eda_pipeline import EDAPipeline

eda_pipe = EDAPipeline()
df = eda_pipe.run(df)

##################################
# PREPROCESSING PIPELINE
##################################
import pipeline_classes.preprocessing as preprocessing
importlib.reload(preprocessing)
from pipeline_classes.preprocessing import DataPreprocessingPipeline

pipeline = DataPreprocessingPipeline()
df = pipeline.fit_transform(df)

##################################
# FEATURE ENGINEERING PIPELINE
##################################

import pipeline_classes.feature_engineering as feature_engineering
importlib.reload(feature_engineering)
from pipeline_classes.feature_engineering import FeatureEngineeringPipeline

feature_pipeline = FeatureEngineeringPipeline()
df = feature_pipeline.fit_transform(df)

####################
# TRAINING PIPELINE
####################
import pipeline_classes.training_testing_pipeline as training_testing_pipeline
importlib.reload(training_testing_pipeline)
from pipeline_classes.training_testing_pipeline import TrainingTestingPipeline

training_pipeline = TrainingTestingPipeline()
rf_model, X_train, X_test, y_train, y_test = training_pipeline.run(df)

############################
# MODEL EVALUATION PIPELINE
############################
import pipeline_classes.model_evaluation_pipeline as model_evaluation_pipeline
importlib.reload(model_evaluation_pipeline)
from pipeline_classes.model_evaluation_pipeline import ModelEvaluationPipeline

evaluation_pipeline = ModelEvaluationPipeline()
evaluation_metrics = evaluation_pipeline.run(rf_model, X_test, y_test)

####################
# MODEL PERSISTENCE
####################
import pipeline_classes.model_persistence_pipeline as model_persistence_pipeline
importlib.reload(model_persistence_pipeline)
from pipeline_classes.model_persistence_pipeline import ModelPersistencePipeline

persistence_pipeline = ModelPersistencePipeline()
persistence_pipeline.run(rf_model, X_train.columns.tolist())

#################
# POST EXECUTION 
#################
utils.open_file("text_files/Post execution.txt")
