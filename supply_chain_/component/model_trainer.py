from supply_chain_.exception import supply_chain_exception
import sys
from supply_chain_.logger import logging
from typing import List
import pandas as pd
import numpy as np
import shutil
import pickle
from supply_chain_.constant import *
from supply_chain_.util.util import *
from supply_chain_.entity.config_entity import *
from supply_chain_.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise supply_chain_exception(e, sys) from e
    
    def start_training_model(self):
        try:
            os.makedirs(self.model_trainer_config.trained_model_file_path_cluster_folder,exist_ok=True)
            #working on training data
            train_df=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            logging.info(f"Reading the file {train_df}")
            test_df=pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
            X_cluster_0_train,y_cluster_0_train,X_cluster_1_train,y_cluster_1_train,X_cluster_0_test,y_cluster_0_test,X_cluster_1_test,y_cluster_1_test=do_train_0(train_df,test_df)
            
            
            X_cluster_0_train.drop(columns = "line_item_value",axis = 1, inplace = True)
            X_cluster_1_train.drop(columns = "line_item_value",axis = 1, inplace = True)
            X_cluster_1_test.drop(columns = "line_item_value",axis = 1, inplace = True)
            X_cluster_0_test.drop(columns = "line_item_value",axis = 1, inplace = True)
            model_cluster1=[]
            model_cluster1_score=[]
            model_cluster2=[]
            model_cluster2_score=[]
            #linear regression for 1st cluster
            lr=LinearRegression()
            lr.fit(X_cluster_0_train,y_cluster_0_train)
            pred=lr.predict(X_cluster_0_train)
            with open("first_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"LinearRegression training score is {r2_score(y_cluster_0_train,pred)}\n")
                f.write(f"LinearRegression testing score is {r2_score(y_cluster_0_test,lr.predict(X_cluster_0_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_0_train,pred)-r2_score(y_cluster_0_test,lr.predict(X_cluster_0_test)))<=0.1:
                model_cluster1.append(lr)
                model_cluster1_score.append(r2_score(y_cluster_0_test,lr.predict(X_cluster_0_test)))
            else:
                model_cluster1.append(lr)
                model_cluster1_score.append(0)
                                            
            #svm
            svm=SVR()
            svm.fit(X_cluster_0_train,y_cluster_0_train)
            pred=svm.predict(X_cluster_0_train)
            with open("first_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"SVM training score is {r2_score(y_cluster_0_train,pred)}\n")
                f.write(f"SVM testing score is {r2_score(y_cluster_0_test,svm.predict(X_cluster_0_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_0_train,pred)-r2_score(y_cluster_0_test,svm.predict(X_cluster_0_test)))<=0.1:
                model_cluster1.append(lr)
                model_cluster1_score.append(r2_score(y_cluster_0_test,svm.predict(X_cluster_0_test)))
            else:
                model_cluster1.append(svm)
                model_cluster1_score.append(0)
            #decision tree
            dr=DecisionTreeRegressor(max_depth=16,random_state=10)
            dr.fit(X_cluster_0_train,y_cluster_0_train)
            pred=dr.predict(X_cluster_0_train)
            with open("first_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"DT training score is {r2_score(y_cluster_0_train,pred)}\n")
                f.write(f"DT testing score is {r2_score(y_cluster_0_test,dr.predict(X_cluster_0_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_0_train,pred)-r2_score(y_cluster_0_test,dr.predict(X_cluster_0_test)))<=0.1:
                model_cluster1.append(dr)
                model_cluster1_score.append(r2_score(y_cluster_0_test,dr.predict(X_cluster_0_test)))
            else:
                model_cluster1.append(dr)
                model_cluster1_score.append(0)
            #randome forest
            rr=RandomForestRegressor(random_state=0)
            rr.fit(X_cluster_0_train,y_cluster_0_train)
            pred=rr.predict(X_cluster_0_train)
            with open("first_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"rr training score is {r2_score(y_cluster_0_train,pred)}\n")
                f.write(f"rr testing score is {r2_score(y_cluster_0_test,rr.predict(X_cluster_0_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_0_train,pred)-r2_score(y_cluster_0_test,rr.predict(X_cluster_0_test)))<=0.1:
                model_cluster1.append(rr)
                model_cluster1_score.append(r2_score(y_cluster_0_test,rr.predict(X_cluster_0_test)))
            else:
                model_cluster1.append(rr)
                model_cluster1_score.append(0)
            
            
            
            #linear regression for 1st cluster
            lr=LinearRegression()
            lr.fit(X_cluster_1_train,y_cluster_1_train)
            pred=lr.predict(X_cluster_1_train)

            with open("second_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"LinearRegression training score is {r2_score(y_cluster_1_train,pred)}\n")
                f.write(f"LinearRegression testing score is {r2_score(y_cluster_1_test,lr.predict(X_cluster_1_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_1_train,pred)-r2_score(y_cluster_1_test,lr.predict(X_cluster_1_test)))<=0.1:
                model_cluster2.append(lr)
                model_cluster2_score.append(r2_score(y_cluster_1_test,lr.predict(X_cluster_1_test)))
            else:
                model_cluster2.append(lr)
                model_cluster2_score.append(0)
            #svm
            svm=SVR()
            svm.fit(X_cluster_1_train,y_cluster_1_train)
            pred=svm.predict(X_cluster_1_train)
            with open("second_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"SVM training score is {r2_score(y_cluster_1_train,pred)}\n")
                f.write(f"SVM testing score is {r2_score(y_cluster_1_test,svm.predict(X_cluster_1_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_1_train,pred)-r2_score(y_cluster_1_test,svm.predict(X_cluster_1_test)))<=0.1:
                model_cluster2.append(svm)
                model_cluster2_score.append(r2_score(y_cluster_1_test,svm.predict(X_cluster_1_test)))
            else:
                model_cluster2.append(svm)
                model_cluster2_score.append(0)
            #decision tree
            dr=DecisionTreeRegressor(max_depth=12,min_samples_split=5,random_state=10)
            dr.fit(X_cluster_1_train,y_cluster_1_train)
            pred=dr.predict(X_cluster_1_train)
            with open("second_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"DT training score is {r2_score(y_cluster_1_train,pred)}\n")
                f.write(f"DT testing score is {r2_score(y_cluster_1_test,dr.predict(X_cluster_1_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_1_train,pred)-r2_score(y_cluster_1_test,dr.predict(X_cluster_1_test)))<=0.1:
                model_cluster2.append(dr)
                model_cluster2_score.append(r2_score(y_cluster_1_test,dr.predict(X_cluster_1_test)))
            else:
                model_cluster2.append(dr)
                model_cluster2_score.append(0)
            #decision tree
            rr=RandomForestRegressor(max_depth=20,min_impurity_decrease=0.2)
            rr.fit(X_cluster_1_train,y_cluster_1_train)
            pred=rr.predict(X_cluster_1_train)
            with open("second_cluster_accuracy_scores.txt","a+") as f:
                f.write(f"rr training score is {r2_score(y_cluster_1_train,pred)}\n")
                f.write(f"rr testing score is {r2_score(y_cluster_1_test,rr.predict(X_cluster_1_test))}\n")
                f.write("---------------------------------------------")
            if abs(r2_score(y_cluster_1_train,pred)-r2_score(y_cluster_1_test,rr.predict(X_cluster_1_test)))<=0.1:
                model_cluster2.append(rr)
                model_cluster2_score.append(r2_score(y_cluster_1_test,rr.predict(X_cluster_1_test)))
            else:
                model_cluster2.append(rr)
                model_cluster2_score.append(0)
                
                
            model_cluster00=model_cluster1[np.argmax(model_cluster1_score)]
            model_cluster11=model_cluster2[np.argmax(model_cluster2_score)]
                
            
            
            

            with open(self.model_trainer_config.trained_model_file_path_cluster0,'wb') as f:
                pickle_file = pickle.dump(model_cluster00,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster0,ROOT_DIR)

            with open(self.model_trainer_config.trained_model_file_path_cluster1,'wb') as f:
                pickle_file = pickle.dump(model_cluster11,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster1,ROOT_DIR)

            return ModelTrainerArtifact(is_trained=True,message="Training has been completed!!",trained_model_file_path_cluster0=self.model_trainer_config.trained_model_file_path_cluster0,trained_model_file_path_cluster1=self.model_trainer_config.trained_model_file_path_cluster1)

        except Exception as e:
           raise supply_chain_exception(e,sys) from e
    
    def initiate_model_training(self):
        self.start_training_model()
