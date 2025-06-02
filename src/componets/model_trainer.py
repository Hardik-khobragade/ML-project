import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor 
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import Custom_exception_handling
from src.utils import save_object,evalute_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainner:
    def __init__(self):
        self.model_trainner_config=ModelTrainerConfig()
        
    def initiate_model_trainner(self,train_array,test_array):
        try:
            logging.info("Model trainner initiate")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "DecisionTree": DecisionTreeClassifier()
            }
            
            params = {
                "LinearRegression": {},
                "RandomForestRegressor": {
                    "n_estimators": [100],
                    "max_depth": [None, 10],
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3, 5],
                },
                "AdaBoostRegressor": {
                    "n_estimators": [50],
                    "learning_rate": [1.0],
                },
                "KNeighborsRegressor": {
                    "n_neighbors": [5],
                    "weights": ["uniform"],
                },
                "XGBRegressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3, 5],
                },
                "CatBoostRegressor": {
                    "iterations": [100],
                    "learning_rate": [0.1],
                    "depth": [3, 6],
                },
                "DecisionTree": {
                    "max_depth": [None, 10],
                    "criterion": ["gini"],
                }
            }
            
            model_report: dict=evalute_model( X_train, y_train, X_test, y_test,models,params)
            
            ##to get best model score in dict
            best_model_score=max(model_report.values())
            
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            
            if best_model_score < 0.6:
                raise Custom_exception_handling("No best model found")
            
            logging.info("Best model found on both train and test dataset")
            
            save_object(
                file_path=self.model_trainner_config.train_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_scorred=r2_score(y_test,predicted)
            
            
            return r2_scorred
            
        except Exception as e:
            raise Custom_exception_handling(e,sys)