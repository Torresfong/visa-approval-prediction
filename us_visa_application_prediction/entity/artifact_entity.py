from dataclasses import dataclass

# artifact entity file is used to define and store the artifact path 
# artifact is the output of each stage of the pipeline which will be used as input for the next stage of the pipeline

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str