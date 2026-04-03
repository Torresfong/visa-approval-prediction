# customise exception for ML project as python's built in exception is too generic 
# custom exception 
# - catch only specific ML failure 
# - Attach structured metadata
# - Produce machine-readable error log for alerting alarm
# - Differentiate retriable and fatal error in retry logic

import traceback # capture full python stack traces as string
from datetime import datetime 
from typing import Any, Optional

class MLBaseException(Exception):
    """
    Root exception for all ML pipeline error
    which all custom exception inherit from this class
    """

    def __init__(
            self,
            message: str,
            stage: Optional[str]=None,
            context: Optional[dict] = None,
            original_exception: Optional[Exception]=None
                 
    ):
        super().__init__(message)
        
        self.message = message
        self.stage = stage or "unknown" # default to unknown if caller doesn't specify stage
        self.context = context or {} # default to empty dictionary which safer for .get() and iteration
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow().isoformat() + "Z" # UTC format with date+time
        self.traceback_str = traceback.format_exc() # capture stack trace at the mon=ment instantiation, capture active exception

    def to_dict(self) -> dict:
        """
        Serialize the exception to dictionary form as log system ingest JSON
        """
# return {} as output is JSON
        return{
            "exception_type": self.__class__.__name__, # like ModelLoadException or DatavalidationException
            "message": self.message,
            "stage":self.stage,
            "context":self.context,
            "timestamp": self.timestamp,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "traceback":self.traceback_str
        }
# return () as output is string object    
    def __str__(self) -> str:
        return(
            f"[{self.__class__.__name__}]"
            f"stage={self.stage}:"
            f"Message={self.message}:"
            f"context={self.context}:"
            f"Time={self.timestamp}"
        )
    
class DatavalidationException(MLBaseException):
    """
    Raise when input data fail validation 
    - Missing require feature columns
    - Wrong number of features (shape missmatch)
    - Non-numeric values in numeric columns
    - Nulls in columns that must not be null
    """

    def __init__(
        self,
        message:str,
        expected_schema: Optional[dict]=None, # what schema we expectedlike (features name,dtype)
        received_schema: Optional[dict]=None, # what schema we actually got 
        invalid_columns:Optional[list]=None, # which specific column raise issue
        **kwargs, # Forward stage,context, like to parent
    ):
        self.expected_schema = expected_schema or {}
        self.received_schema = received_schema or {}
        self.invalid_columns = invalid_columns or []

        # Merge validation details into context so they appear in to_dict() log automatically
        context = kwargs.pop("context",{})
        context.update({
            "expected_schema": self.expected_schema,
            "received_schema": self.received_schema,
            "invalid_column": self.invalid_columns, 
        })

        # Call super().__init__() last in a child __init__ when you need to prepare or transform data before handing it to the parent.
        # only call super().__init__() first when  parent sets up something the child immediately needs — which is rare in exception classes.
        super().__init__(message,context=context,**kwargs)
    
class DataIngestionException(MLBaseException):
    """
    Raise when data cannot be fetch or read from source
    - database connection timeout,s3 file not found,csv parsing error
    """
    def __init__(
        self,
        message:str,
        data_source:Optional[str]=None,
        **kwargs
    ):
        self.data_source = data_source

        context =kwargs.pop("context",{})
        context["data_source"]=data_source # inclde source path in every log entry
        
        super().__init__(message, context=context,**kwargs)
    
class FeatureEngineeringException(MLBaseException):
    """
    Raise when feature tranformation or encoding step fail
    - Unseen category label during one-hot encoding
    - Nan introduced during feature computation
    """
    def __init__(
            self,
            message:str,
            feature_name:Optional[str]=None,  # which feature failed
            transformation:Optional[str]=None, # which transformation failed
            **kwargs
    ):
        self.feature_name =feature_name
        self.transformation = transformation

        context = kwargs.pop("context",{})
        context.update({
            "feature_name": feature_name,
            "transformation" : transformation
        })

        super().__init__(message, context=context,**kwargs)

class ModelLoadException(MLBaseException):
    """
    Raise when a model artifact cannot be loaded from disk or registry
    - Model file not found at expected path
    - Pickle/ONNX file is corrupted
    - Version mismatch btw model and serving data
    - Mlflow/model registry API failure
    """
    def __init__(
        self, 
        message: str,
        model_name: Optional[str]=None,
        model_version:Optional[str]=None,
        model_path: Optional[str]=None,
        **kwargs
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_path = model_path

        context = kwargs.pop("context",{})
        context.update({
            "model_name" : model_name,
            "model_version" : model_version,
            "model_path" : model_path
        })
        super().__init__(message,context=context,**kwargs)

class ModelPredictionException(MLBaseException):
    """
    Raised when the model.predict or model.predict_proba() call fail at runtime
    - Input tensor shape doesn't match model's expected input
    - GPU out-of-memory error during batch inference
    - NaN produced in model output (numerical instability)
    - ONNX runtime execution failure
    """
    def __init__(
        self,
        message: str,
        model_name: Optional[str]= None,
        input_shape: Optional[tuple] = None,
        batch_size : Optional[int] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.input_shape = input_shape 
        self.batch_size = batch_size
        
        context = kwargs.pop("context",{})
        context.update({
            "model_name": model_name,
            "input_shape": str(input_shape) if input_shape else None, # Convert tuple to string for JSON
            "batch_size" : batch_size
        })

        super().__init__(message, context=context, **kwargs)

class ConfigurationException(MLBaseException):
    """
    Raised when required configuratio is missing,invalid or incompatible
    - Missing environment variable (MODEL_REGISTRY_URL not set)
    - Threshold value outside [0, 1]
    - Incompatible hyperparameter combination
    """
    def __init__(
        self,
        message: str,
        config_key: Optional[str]= None,
        config_value: Optional[Any] = None,
        **kwargs  
    ):
        self.config_key = config_key
        self.config_value = config_value

        context = kwargs.pop("context",{})
        context.update({
            "config_key" : config_key,
            "config_value" : str(config_value) if config_value is not None else None,
        })

        super().__init__(message, context = context, **kwargs)

# =============================================================================
# INFRASTRUCTURE EXCEPTIONS
# Raised for external service/resource failures outside the ML code itself.
# =============================================================================

class ModelRegistryException(MLBaseException):
    """
    Raised when interaction with a model registry (MLflow, W&B, SageMaker) fails.
    Separate from ModelLoadException because the model file itself may be fine,
    but the registry API is down.
    """
    pass  # Inherits all behavior from MLBaseException; specialization is in the name


class DatabaseException(MLBaseException):
    """
      Raised when database reads/writes for prediction logging or feature retrieval fail.
    """
    pass


class ExternalAPIException(MLBaseException):
    """
    Raised when a third-party API call (feature store, enrichment service) fails.
    """

    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,  # Which endpoint was called
        status_code: Optional[int] = None,   # HTTP status code returned
        **kwargs,
    ):
        self.api_endpoint = api_endpoint
        self.status_code = status_code

        context = kwargs.pop("context", {})
        context.update({
            "api_endpoint": api_endpoint,
            "status_code": status_code,
        })

        super().__init__(message, context=context, **kwargs)