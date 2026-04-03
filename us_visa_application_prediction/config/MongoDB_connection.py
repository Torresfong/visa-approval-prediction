import sys
from us_visa_application_prediction.exception import *
from us_visa_application_prediction.logger import logging
import os
from us_visa_application_prediction.constants import DATABASE_NAME, MONGODB_URL_KEY
import pymongo
import certifi

ca = certifi.where()

class MongoDBClient:
    """
    Description :   This class is responsible for connecting to MongoDB database and performing database operations. 
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca) # tlsCAFile=ca is used to specify the path to the CA certificate file for secure connection
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise DatabaseException(f"Error while connecting to MongoDB: {str(e)}") from e # from e is called exception chaining used to get the full stack trace of the original exception along with the new exception. It helps in debugging and understanding the root cause of the error.