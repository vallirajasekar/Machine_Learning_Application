import yaml
import os
import pandas as pd

def write_schema_yaml(csv_file):
    df = pd.read_csv(csv_file)
    
    num_columns = len(df.columns)
    columns_names = df.columns.tolist()
    columns_dtypes = df.dtypes.astype(str).tolist()
    
    ## Create schema Dictionary
    schema = {
        "filename": os.path.basename(csv_file),
        "Numberofcolumns": num_columns,
        "ColumnNames": dict(zip(columns_names, columns_dtypes))
    }
    
    # Write schema to schema.yaml file
    ROOT_DIR = os.getcwd()
    SCHEMA_PATH = os.path.join(ROOT_DIR, 'config', 'schema.yaml')
    
    with open(SCHEMA_PATH, "w") as file:
        yaml.dump(schema, file)
