import uuid
from src.exception import ApplicationException
from typing import List
from multiprocessing import Process
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
import sys





class Pipeline():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config

            
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> dataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))
            logging.info('Data Ingestion Starts here')
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
    def run_pipeline(self):
        try:
             #data ingestion
            dataIngestionArtifact = self.start_data_ingestion()
           
            # data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            
            # model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
          
         
        except Exception as e:
            raise ApplicationException(e, sys) from e

        