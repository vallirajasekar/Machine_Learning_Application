#from pymongo.mongo_client import MongoClient
import pandas as pd 
import pymongo
import json

client= "mongodb+srv://sekar:btmsekar@cluster0.vwolx5r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATA_FILE_PATH=(r"/Users/vallirajasekar/Desktop/ML/Machine_Learning_Application/Data/train.csv")
DATABASE="Machine_learning"
COLLECTION_NAME="DATASET"

if __name__=="__main__":
    df=pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Colums of our Data:{df.shape}")
    df.reset_index(drop=True,inplace=True)
    json_record=list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    client[DATABASE][COLLECTION_NAME].insert_many(json_record)
    
    


# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
