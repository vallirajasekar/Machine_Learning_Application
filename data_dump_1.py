from pymongo import MongoClient
import pandas as pd 
import pymongo
import json
import os,sys
from schema import write_schema_yaml

uri= "mongodb+srv://sekar:btmsekar@cluster0.vwolx5r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATA_FILE_PATH=(r"/Users/vallirajasekar/Desktop/ML/Machine_Learning_Application/Data/train.csv")
DATABASE="Machine_learning"
COLLECTION_NAME="DATASET"

if __name__=="__main__":
    
    ROOT_DIR=os.getcwd()
    
    DATA_FILE_PATH=os.path.join(ROOT_DIR,'Data','train.csv')
    
    FILE_PATH=os.path.join(ROOT_DIR,DATA_FILE_PATH)
    
    write_schema_yaml(csv_file=DATA_FILE_PATH)
    
    df=pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns of our Data:{df.shape}")
     # convert dataframe to the list of dictionaries (JSON RECORDS)
    json_records=json.loads(df.to_json(orient="records"))
    print(json_records[0])
    
    client=pymongo.MongoClient(uri)
    db=client[DATABASE]
    collection=db[COLLECTION_NAME]
    collection.insert_many(json_records)
    client.close()

    
    
    
    # #df.reset_index(drop=True,inplace=True)
    # json_record=list(json.loads(df.T.to_json()).values())
    # print(json_record)
    # client[DATABASE][COLLECTION_NAME].insert_many(json_record)
    
    # # # convert dataframe to the list of dictionaries (JSON RECORDS)
    # # json_records=json.loads(df.to_json(orient="records"))
    # # print(json_records[0])
    
    # # # client=pymongo.MongoClient(uri)
    # # # db=client[DATABASE]
    # # # collection=db[COLLECTION_NAME]
    # # # collection.insert_many(json_records)
    # # # client.close()

    
    
    