import os

ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)
print(ROOT_DIR)
print(CONFIG_DIR)
print(SCHEMA_FILE)
print(CONFIG_FILE_PATH)