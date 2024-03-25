from pymongo import MongoClient
import pandas as pd 
import pymongo
import json

uri= "mongodb+srv://sekar:btmsekar@cluster0.vwolx5r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATA_FILE_PATH=(r"/Users/vallirajasekar/Desktop/ML/Machine_Learning_Application/Data/train.csv")
DATABASE="Machine_learning"
COLLECTION_NAME="DATASET"

if __name__=="__main__":
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

    
    


# # Create a new client and connect to the server
# client = MongoClient(uri)
# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)
