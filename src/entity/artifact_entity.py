from dataclasses import dataclass

@dataclass
class dataIngestionArtifact:
    train_file_path:str
    test_file_path:str
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    train_target_file_path:str
    transformed_test_file_path:str
    test_target_file_path:str
    feature_engineering_object_file_path: str
    
