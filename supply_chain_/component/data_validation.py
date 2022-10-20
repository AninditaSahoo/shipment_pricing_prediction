from supply_chain_.logger import logging
from supply_chain_.exception import supply_chain_exception
from supply_chain_.entity.config_entity import DataValidationConfig
from supply_chain_.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys
import pandas  as pd
from supply_chain_.constant import *
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from supply_chain_.util.util import *
import json

class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.timestamp=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        except Exception as e:
            raise supply_chain_exception(e,sys) from e


    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False
            #validation check for train and test file
            df=pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)
            l=[]
            for cols in df.columns:
                yaml=read_yaml_file(SCHEMA_FILE_PATH)
                if cols in yaml.keys():
                    if str(df[cols].dtype)==yaml[cols]:
                        l.append(True)
            if len(l)==len(yaml.keys()):
                validation_status=True
            os.makedirs(self.data_validation_config.raw_data_path,exist_ok=True)
            df.to_csv(os.path.join(self.data_validation_config.raw_data_path,"validated_data.csv"),index=False)
            return validation_status
        except Exception as e:
            raise supply_chain_exception(e,sys) from e

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            status=self.validate_dataset_schema()
            data_validation_artifact = DataValidationArtifact(validate_csv_path=os.path.join(self.data_validation_config.raw_data_path,"validated_data.csv")
                ,is_validated=status,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise supply_chain_exception(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")