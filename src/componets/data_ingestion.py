import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
from src.exception import Custom_exception_handling
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

from src.componets.data_transformation import DataTransformation
from src.componets.data_transformation import DataTransformationConfig

from src.componets.model_trainer import ModelTrainerConfig
from src.componets.model_trainer import ModelTrainner




@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df=pd.read_csv('data/stud.csv')
            logging.info("Read the data from dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initiate")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data ingestion is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise Custom_exception_handling(e,sys)
            
          
            
if __name__ =="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_trasformaton=DataTransformation()
    train_arr,test_arr,_=data_trasformaton.initiate_data_transformation(train_data,test_data)
    
    model_trainner=ModelTrainner()
    print(model_trainner.initiate_model_trainner(train_arr,test_arr))