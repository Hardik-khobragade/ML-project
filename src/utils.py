import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
from src.exception import Custom_exception_handling
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise Custom_exception_handling(e, sys)
            
def evalute_model(X_train,y_train,X_test,y_test,models,params):
    try:
        
        report={}
        
        for i in range (len(list(models))):
            model = list(models.values())[i]
            para= list(params.values())[i]
            
            grid_cv=GridSearchCV(model,para,cv=3)
            grid_cv.fit(X_train,y_train)
            
            model.set_params(**grid_cv.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(y_train_pred,y_train)
            test_model_score=r2_score(y_test_pred,y_test)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise Custom_exception_handling(e, sys)      

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Custom_exception_handling(e, sys)    
    
