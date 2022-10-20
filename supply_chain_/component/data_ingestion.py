from supply_chain_.entity.config_entity import DataIngestionConfig
import sys,os
from supply_chain_.exception import supply_chain_exception
from supply_chain_.constant import *
from supply_chain_.logger import logging
from supply_chain_.entity.artifact_entity import DataIngestionArtifact
import numpy as np
import pandas as pd
from supply_chain_.util.util import *
from sklearn.model_selection import train_test_split
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
class DataIngestion:
    
    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
            os.makedirs(self.data_ingestion_config.raw_data_dir,exist_ok=True)
        except Exception as e:
            raise supply_chain_exception(e,sys)
    def get_data_from_database(self):
        try:
            cloud_config= {'secure_connect_bundle': 'E:\Ineuron\class\secure-connect-supply-chain (1).zip'}
            auth_provider = PlainTextAuthProvider('eEAtAbOLSLKGTBDWaHIxyAZF', 'i6caDSku2Zcg9hLPdqY7Ze+uD2w_aGmSM7bx-MK-moEjxtE-BZNTfDkEOHqvp_wNRlIWo-BL1Zvh1p_7xY.fxXpBZ6SFQMnDi9+tqXNlJ.Bk285gs+MeZHvRj+f4P-oX')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect("supply_chain")

            # row = session.execute("SELECT * FROM forest_cover.train").one()
            session.row_factory = pandas_factory
            session.default_fetch_size = None
            query = "SELECT * FROM supply_chain.supply_data_ineuron_ai_database_proj"
            rslt = session.execute(query, timeout=None)
            df = rslt._current_rows
            df.to_csv(os.path.join(self.data_ingestion_config.raw_data_dir,"dumped_data.csv"),index=False)
        except Exception as e:
            raise supply_chain_exception(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.get_data_from_database()
            return DataIngestionArtifact(raw_data_file_path=os.path.join(self.data_ingestion_config.raw_data_dir,"dumped_data.csv"),is_ingested=True,message="Data Ingestion done")
        except Exception as e:
            raise supply_chain_exception(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")



