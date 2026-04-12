import os
from datetime import date

# constant file name use to store fixed values, configuration parameters, paths
# Centralize the setting, improve code readability,maintainability, and reusability

DATABASE_NAME = "us_visa"  # Naming convention for constant variable is all uppercase with underscores separating words. This makes it clear that the value is a constant and should not be changed throughout the codebase.

COLLECTION_NAME = "visa_collection"

MONGODB_URL_KEY = "MONGODB_URL_KEY"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
FILE_NAME:str = "EasyVisa.csv"
MODEL_FILE_NAME = "model.pkl"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILE_PATH = os.path.join(
    ROOT_DIR,
    "config",
    "schema.yaml"
)

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing_object.pkl"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_collection" # collection name in mongodb where we will store the data after data ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion" # data ingestion directory name inside artifact directory and then create data ingestion directory with timestamp inside data ingestion directory
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store" # feature store is a directory where we will store the data after data ingestion in the form of csv file and then we will use this file for data validation and data transformation
DATA_INGESTION_INGESTED_DIR: str = "ingested" # ingested directory is a directory where we will store the data after data ingestion in the form of csv file and then we will use this file for data validation and data transformation
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2



"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"



"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")



MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "usvisabucket25"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 8080