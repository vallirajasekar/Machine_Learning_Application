import os
import pymongo
from dotenv import load_dotenv

def mongo_client():
    ROOT_DIR = os.getcwd()
    env_file_path = os.path.join(ROOT_DIR, '.env')
    load_dotenv(dotenv_path=env_file_path)
    
    USER_NAME = os.getenv("USER_NAME")
    PASSWORD = os.getenv("PASSWORD")
    CLUSTER_NAME = os.getenv("CLUSTER_LEVEL")
    
    mongo_db_url = f"mongodb+srv://{USER_NAME}:{PASSWORD}@{CLUSTER_NAME}.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    try:
        client = pymongo.MongoClient(mongo_db_url)
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        return None
