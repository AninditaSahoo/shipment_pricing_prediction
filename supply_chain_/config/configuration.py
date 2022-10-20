from supply_chain_.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,TrainingPipelineConfig
from supply_chain_.logger import logging
from supply_chain_.exception import supply_chain_exception
from supply_chain_.constant import *
from supply_chain_.util.util import *
import os,sys

class Configuration:
    def __init__(self,config_file_path:str=CONFIG_FILE_PATH,current_time_stamp:str=CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info=read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config=self.get_training_pipeline_config()
            self.timestamp=current_time_stamp
        except Exception as e:
            raise supply_chain_exception(e,sys) from e
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_info=self.config_info[DATA_INGESTION_CONFIG_KEY]
            artifact_dir=self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir=os.path.join(artifact_dir,DATA_INGESTION_ARTIFACT_DIR,self.timestamp)
            raw_data_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ingested_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
 
          
            data_ingestion_config=DataIngestionConfig(
            raw_data_dir=raw_data_dir, 
)
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config

        except Exception as e:
            raise supply_chain_exception(e,sys) from e


    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info=self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            artifact_dir = self.training_pipeline_config.artifact_dir
     

            data_validation_data_dir=os.path.join(
                artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.timestamp,DATA_VALIDATION_RAW_DIR_NAME
            )
            

            data_validation_config = DataValidationConfig(
                raw_data_path=data_validation_data_dir
            )
            return data_validation_config
        except Exception as e:
            raise supply_chain_exception(e,sys) from e


    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_config_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transformation_artifact_dir=os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.timestamp
            )



            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )

            preprocessed_object_folder_path=os.path.join(
                data_transformation_artifact_dir,
                data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])

            
            transformed_train_dir=os.path.join(
            data_transformation_artifact_dir,
            data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
            data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            )


            transformed_test_dir = os.path.join(
            data_transformation_artifact_dir,
            data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY],
            data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]

            )
            

            data_transformation_config=DataTransformationConfig(
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessed_object_folder_path=preprocessed_object_folder_path
            )

            logging.info(f"Data transformation config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise supply_chain_exception(e,sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_trainer_artifact_dir=os.path.join(artifact_dir,MODEL_TRAINER_ARTIFACT_DIR,self.timestamp)
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            trained_model_file_path_cluster_folder=os.path.join(model_trainer_artifact_dir,model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY])

           

            trained_model_file_path_cluster0 = os.path.join(model_trainer_artifact_dir,model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY_0]
            )

            trained_model_file_path_cluster1 = os.path.join(model_trainer_artifact_dir,model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY],model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY_1]
            )
        

            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            model_trainer_config = ModelTrainerConfig(trained_model_file_path_cluster_folder=trained_model_file_path_cluster_folder,trained_model_file_path_cluster0=trained_model_file_path_cluster0,trained_model_file_path_cluster1=trained_model_file_path_cluster1,base_accuracy=base_accuracy)
            logging.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise supply_chain_exception(e,sys) from e




    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config=self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir=os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
            training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise supply_chain_exception(e,sys) from e