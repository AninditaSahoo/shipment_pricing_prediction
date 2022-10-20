import pickle
import shutil
from sklearn import preprocessing
from supply_chain_.exception import supply_chain_exception
from supply_chain_.logger import logging
from supply_chain_.entity.config_entity import DataTransformationConfig 
from supply_chain_.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
from sklearn.model_selection import train_test_split
import numpy as np
from supply_chain_.constant import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
import pandas as pd
from supply_chain_.constant import *
from supply_chain_.util.util import *
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns



class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise supply_chain_exception(e,sys) from e
    def get_transformer_object_and_test_train_dir(self):
        try:
            os.makedirs(self.data_transformation_config.transformed_train_dir,exist_ok=True)
            logging.info(f"Train directory created-->{self.data_transformation_config.transformed_train_dir}")
            os.makedirs(self.data_transformation_config.transformed_test_dir,exist_ok=True)
            logging.info(f"Test directory created-->{self.data_transformation_config.transformed_test_dir}")
            os.makedirs(self.data_transformation_config.preprocessed_object_folder_path,exist_ok=True)
            logging.info(f"Preprocessing folder path created {self.data_transformation_config.preprocessed_object_folder_path}")
            #scaling on train data
            df=pd.read_csv(self.data_validation_artifact.validate_csv_path)
            pd.set_option('display.max_columns', None)
            print(df.head())
            logging.info(f"Reading the train file path which is {df}")
            for rows in range(df.shape[0]):
                if df.iloc[rows,3] not in (df["country"].value_counts().index[:11]):
                    df.iloc[rows,3]="Others"
            logging.info("Country column handled")
            

            logging.info("vendor_inco_term column handled")
            df["vendor_inco_term"]=df["vendor_inco_term"].str.replace("N/A - From RDC","NA")
            for rows in range(df.shape[0]):
                if (df.loc[rows,"fulfill_via"]=="From RDC") & (df.loc[rows,"shipment_mode"]=="MIssing"):
                    df.loc[rows,"shipment_mode"]=df[df["fulfill_via"]=="From RDC"]["shipment_mode"].mode()[0]
            
            df=df[df["shipment_mode"]!="MIssing"]

            df=df[df["pq_first_sent_to_client_date"]!="Date Not Captured"]

            df=df.reset_index(drop=True)
            df["po_sent_to_vendor_date"]=df["po_sent_to_vendor_date"].str.replace("N/A - From RDC","0")

            df=df[df["po_sent_to_vendor_date"]!="Date Not Captured"]

            df=df.reset_index(drop=True)

            df.drop("vendor",axis=1,inplace=True)
            df.drop("item_description",axis=1,inplace=True)
            for i in range(df.shape[0]):
                if df.loc[i,"brand"] not in df["brand"].value_counts().index[:5]:
                    df.loc[i,"brand"]="Others"

            test_data3=df["dosage"].str.split("/",expand=True)
            test_data3.columns=["first","second","third"]
            df["dosage"]=test_data3["first"]
            test_data4=df["dosage"].str.split("mg",expand=True)
            test_data4.columns=["first","second"]
            df["dosage"]=test_data4["first"]
            df["dosage"]=df["dosage"].str.replace("MIssing","0")
            test_data5=df["dosage"].str.split("+",expand=True)
            test_data5=test_data5.fillna(0)
            test_data5.columns=["first","second"]
            test_data5=test_data5[test_data5["first"]!="2g"]
            test_data5["add"]=pd.to_numeric(test_data5["first"])+pd.to_numeric(test_data5["second"])
            df=df[df["dosage"]!="2g"]
            df=df.reset_index(drop=True)
            test_data5=test_data5.reset_index(drop=True)
            df["dosage"]=test_data5["add"]
            pd.set_option('display.max_columns', None)
       
            for i in range(df.shape[0]):
                if df.loc[i,"dosage_form"] not in df["dosage_form"].value_counts().index[:5]:
                    df.loc[i,"dosage_form"]="Others"

            def seperate_weight(x):
                if "See" in x:
                    y=x.split(":")
                    z=y[1].split(")")
                    value=df[df["id"]==int(z[0])]["weight"].index[0]
                    return value
                return x
            
            df["weight"]=df["weight"].str.replace("Weight Captured Separately","0")

            df["weight"]=df["weight"].apply(lambda z:seperate_weight(z))

            df["weight"]=pd.to_numeric(df["weight"])

            df["freight_cost"]=df["freight_cost"].str.replace("Freight Included in Commodity Cost","0")

            def seperate_freight(x):
                if "See" in x:
                    y=x.split(":")
                    z=y[1].split(")")
                    value=df[df["id"]==int(z[0])]["freight_cost"].index[0]
                    return value
                return x

            df["freight_cost"]=df["freight_cost"].apply(lambda z:seperate_freight(z))

            for i in range(df.shape[0]):
                if df.loc[i,"freight_cost"]=="Invoiced Separately":
                    df.loc[i,"freight_cost"]=0

            df["freight_cost"]=pd.to_numeric(df["freight_cost"])

            df["line_item_insurance"]=df["line_item_insurance"].str.replace("MIssing","0")

            df["line_item_insurance"]=pd.to_numeric(df["line_item_insurance"])

            df["shipping_price"]=df["line_item_value"]+df["pack_price"]+df["freight_cost"]

            df=df[df["shipping_price"]!=0]

            df=df.reset_index(drop=True)

            df.drop(["id","project_code","asn_dn","delivered_to_client_date","delivery_recorded_date","scheduled_delivery_date","vendor_inco_term","pq_first_sent_to_client_date","po_so","pq","po_sent_to_vendor_date","manufacturing_site","molecule_test_type"],axis=1,inplace=True)


            df=outlier_detector(df,"freight_cost")

            df=df.reset_index(drop=True)
            df1=df.drop("shipping_price",axis=1)


            #creating pipeline
            numeric_processor=Pipeline(steps=[("imputation_constant",SimpleImputer(missing_values=np.nan,fill_value=0)),("scaler",StandardScaler())])
            
            categorical_processor=Pipeline(steps=[("onehot",OneHotEncoder(drop="first"))])

            preprocessor=ColumnTransformer([("categorical",categorical_processor,["brand","country","dosage_form","first_line_designation","fulfill_via","managed_by","product_group","shipment_mode","sub_classification"]),("numerical",numeric_processor,["dosage","freight_cost","line_item_insurance","line_item_quantity","line_item_value","pack_price","unit_of_measure","unit_price","weight"])])


            df2=pd.DataFrame(preprocessor.fit_transform(df1),columns=['brand_Uni-Gold','brand_Others',"brand_Kaletra","brand_Determine","brand_Aluvia",'country_Others', 'country_Mozambique', 'country_South Africa', 'country_CÃte dIvoire', 'country_Nigeria', 'country_Zimbabwe', 'country_Uganda', 'country_Vietnam', 'country_Rwanda', 'country_Haiti', 'country_Tanzania','dosage_form_Tablet - FDC', 'dosage_form_Test kit', 'dosage_form_Oral solution', 'dosage_form_Capsule', 'dosage_form_Others','first_line_designation_Yes','fulfill_via_Direct Drop','managed_by_Haiti Field Office', 'managed_by_Ethiopia Field Office','product_group_HRDT', 'product_group_ANTM', 'product_group_MRDT', 'product_group_ACT','shipment_mode_Air', 'shipment_mode_Ocean', 'shipment_mode_Air Charter','sub_classification_HIV test', 'sub_classification_Pediatric', 'sub_classification_Malaria', 'sub_classification_HIV test - Ancillary', 'sub_classification_ACT',"dosage","freight_cost","line_item_insurance","line_item_quantity","line_item_value","pack_price","unit_of_measure","unit_price","weight"])



            df3=pd.concat([df2,df["shipping_price"]],axis=1)
            df_train,df_test=train_test_split(df3,test_size=0.2,random_state=0)

            df_train=df_train.reset_index(drop=True)
            df_test=df_test.reset_index(drop=True)


            df_train.to_csv(os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),index=False)


            df_test.to_csv(os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),index=False)

            path_pkl=self.data_transformation_config.preprocessed_object_file_path
            with open(path_pkl,"wb") as f:
                pickle.dump(preprocessor,f)
            shutil.copy(path_pkl,ROOT_DIR)
            return DataTransformationArtifact(is_transformed=True,message="Data Transformed",transformed_train_file_path=os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),transformed_test_file_path=os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path)

        except Exception as e:
            raise supply_chain_exception(e,sys) from e

    def initiate_data_transformation(self):
        try:
            data_transformation_artifact=self.get_transformer_object_and_test_train_dir()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise supply_chain_exception(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")


