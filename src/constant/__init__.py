import os,sys

FILE_NAME="data.csv"


ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)


## Varaiable related to Data Ingestion Pipeline 

DATA_INGESTION_CONFIG_KEY="data_ingestion_config"
DATA_INGESTION_DATABASE_NAME="data_base"
DATA_INGESTION_COLLECTION_NAME="collection_name"
DATA_INGESTION_ARTIFACT_DIR="data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY="raw_data_dir"
DATA_INGESTION_INGESTED_DIR_KEY="ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY="ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY="ingested_test_dir"
CONFIG_FILE_KEY="config"

# Schema file path 
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='schema.yaml'
SCHEMA_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)

## Varaiable related to Data Transformation Pipeline
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
TRANSFORMATION_FILE='transformation.yaml'
TRANSFORMATION_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TRANSFORMATION_FILE)
TARGET_COLUMN_KEY= 'target_column'
NUMERICAL_COLUMN_KEY= 'numerical_columns'
CATEGORICAL_COLUMNS ='categorical_columns'
DROP_COLUMNS= 'drop_columns'
SCALING_COLUMNS='scaling_columns'
OUTLIER_COLUMNS='outlier_columns'

# key  ---> config.yaml---->values
#Varaible related to Data Transformation Pipeline 
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION='data_transformation_dir'
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_FEA_ENG_FILE_NAME_KEY='feature_eng_file'
DATA_TRANSFORMATION_PREPROCESSOR_NAME_KEY='preprocessed_object_file_name'

PIKLE_FOLDER_NAME_KEY = "prediction_files"



