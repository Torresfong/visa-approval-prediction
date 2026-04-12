from us_visa_application_prediction.config.MongoDB_connection import MongoDBClient
from us_visa_application_prediction.constants import DATABASE_NAME
from us_visa_application_prediction.exception import DataIngestionException
import pandas as pd
import sys
from typing import Optional
import numpy as np



class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
    
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise DataIngestionException(f"Error while connecting to MongoDB: {str(e)}") from e
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collection as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                if self.mongo_client.client is None:
                    raise DataIngestionException("MongoDB client is not initialized.")
                collection = self.mongo_client.client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "case_id" in df.columns.to_list():
                df = df.drop(columns=["case_id"])
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise DataIngestionException(f"Error while exporting collection{collection_name} as dataframe:")