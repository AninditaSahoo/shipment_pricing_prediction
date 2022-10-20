import yaml
from supply_chain_.exception import supply_chain_exception
import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # Linear Regression from STATSMODEL
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise supply_chain_exception(e,sys) from e


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


def outlier_detector(dataframe,column_name):

    q1=dataframe[column_name].quantile(0.25)
    q3=dataframe[column_name].quantile(0.75)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    return dataframe[(dataframe[column_name]>=lower_bound)&(dataframe[column_name]<=upper_bound)]

def do_train_0(train_dataframe,test_dataframe):
    try:
        #training
        X=train_dataframe.drop("shipping_price",axis=1)
        y=train_dataframe["shipping_price"]


        kmeans=KMeans(n_clusters=2,init="k-means++",random_state=0)
        labels=kmeans.fit_predict(X)
        cluster_df=pd.concat([X,y,pd.DataFrame(labels,columns=["cluster"])],axis=1)
        # K = range(1,10)
        # for num_clusters in list(K):
        #     kmeans = KMeans(n_clusters=num_clusters, init = "k-means++")
        #     preds=kmeans.fit_predict(X)
        #     score = silhouette_score(X, preds)
        #     with open("silhoute_score.txt","a+") as f:
        #         f.write("For n_clusters = {}, silhouette score is {}".format(num_clusters, score)+'\n')
                
        df1=cluster_df[cluster_df["cluster"]==0]
        df2=cluster_df[cluster_df["cluster"]==1]
        df1.drop("cluster",axis=1,inplace=True)
        X=df1.drop("shipping_price",axis=1)
        y=df1["shipping_price"]
        df2.drop("cluster",axis=1,inplace=True)
        XX=df2.drop("shipping_price",axis=1)
        yy=df2["shipping_price"]
        #testing
        X1=test_dataframe.drop("shipping_price",axis=1)
        labels1=kmeans.predict(X1)
        cluster_df_valid=pd.concat([X1,test_dataframe["shipping_price"],pd.DataFrame(labels1,columns=["cluster"])],axis=1)
        df1_valid=cluster_df_valid[cluster_df_valid["cluster"]==0]
        df1_valid.drop("cluster",axis=1,inplace=True)
        X1=df1_valid.drop("shipping_price",axis=1)
        y1=df1_valid["shipping_price"]
        df2_valid=cluster_df_valid[cluster_df_valid["cluster"]==1]
        df2_valid.drop("cluster",axis=1,inplace=True)
        XX1=df2_valid.drop("shipping_price",axis=1)
        yy1=df2_valid["shipping_price"]
        
        
        
        return X,y,XX,yy,X1,y1,XX1,yy1
    except Exception as e:
        raise supply_chain_exception(e,sys) from e