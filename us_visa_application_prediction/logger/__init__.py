import logging 
from logging.handlers import RotatingFileHandler # prevent logfile from infinity
from typing import Any, Optional, Callable, cast
import os  ## To read LOG_LEVEL and LOG_FORMAT from environment variables
import sys # to direct logs to stdout(for container)
import json # for serializing structured log record to JSON string
import time # to measure elapsed time (prediction latency)
import uuid # For generating unique request/prediction IDs
from datetime import datetime 
from functools import wraps # preserve original function name/docstring in decorators


"""
Logging is like project diary that record everything happen so programmer can investigate later

why Structured logging for ML 
- Plain text logs ("ERROR: model failed") are hard to query in production
- Structured JSON logs let you filter by model_name, stage, run_id, etc.
  in tools like Kibana, Datadog, CloudWatch Insights, or Splunk
- Every log entry carries consistent fields, making automated alerting trivial
- Performance metrics (latency, throughput) can be extracted without regex
"""

class MLLogRecord(logging.LogRecord):
# Define LogRecord type that officailly declare ml_context 
    ml_context = Optional[dict]
    # now type checker know this fields exist

logging.setLogRecordFactory(MLLogRecord)

class JSONFormatter(logging.Formatter):
    # convert each Logrecord into JSON object
    # replace python default "INFO:module:message" format with
    # {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", ...}

    def __inti__(self, service_name:str = "ml-classifier", environment:str = "production"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:        
        try:
            """
            Called automatic for each log record before write to handle
            `record` is a LogRecord object with field: levelname, name, getMessage(), etc.
            We use try and error to avoid crashing by getting fallback of some useful back even formatter itself have bug
            """
            # Build core log dictionary
            log_data = {
                "timestamp" : datetime.utcfromtimestamp(record.created).isoformat() + "Z",
                "level" : record.levelname,         # "DEBUG", "INFO", "WARNING", etc
                "logger" : record.name,             
                "message" : record.getMessage(),    # Actual log message string
                "service" : self.service_name,      
                "environment" : self.environment,   # "production","staging","development"
                "module" : record.module,           # python module filename 
                "function" : record.funcName,       # Name of the funciton that call the logger
                "line" : record.lineno,             # line no in sourec file
                }

            ml_record = cast(MLLogRecord,record)
            ml_context = getattr(ml_record, "ml_context",None) # Try to get "ml_context"  which is extra from record, if doest exist just give none = no cracs no type error
            if isinstance(ml_context,dict):
                log_data.update(ml_context)
                # merge the extra metadata into main log dictionary
                # eg: #   logger.info("Prediction made", extra={"ml_context": {"model": "v3", "score": 0.91}})

            # Handling error and exception
            if record.exc_info:
            # automatically filled when error occur
                exc_type, exc_value, _ = record.exc_info   # unpack the 3-tuple cleanly
                log_data["exception"] = {
                    "type": exc_type.__name__ if exc_type is not None else "UnknownError",
                    "message": str(exc_value) if exc_value is not None else "No message",
                    "traceback": self.formatException(record.exc_info),
                    # formatException converts (type, value, tb) tuple into a readable traceback string
            }
            
            return json.dumps(log_data, separators = (",",":"),default =str )
            # Seperator=(",",":") produce compact JSON with no space to save log storage
            # default =str handles non JSON serializable type 
            # by converting them to string better than crashing the logger

        except Exception as e:
            
            return json.dumps({
                "level" : "ERROR",
                "message" : f"LOG formatting failed:{e}",
                "original_message" : record.getMessage(),
            })

class ColoredConsoleFormatter(logging.Formatter):
    """
    Adds ANSI color codes to log levels for terminal readability.
    These color codes are invisible noise in JSON log aggregators,
    so this formatter is only used when LOG_FORMAT=console (local dev).
    """

    # ANSI escape codes: \033[<code>m sets terminal color, \033[0m resets it
    COLORS = {
        "DEBUG":    "\033[36m",    # Cyan   — low-priority diagnostic info
        "INFO":     "\033[32m",    # Green  — normal operation events
        "WARNING":  "\033[33m",    # Yellow — something unexpected but non-fatal
        "ERROR":    "\033[31m",    # Red    — a real failure occurred
        "CRITICAL": "\033[35m",    # Magenta — system is in a dangerous state
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")  # Default to no color for unknown levels
        timestamp = datetime.utcfromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Pad level name to 8 chars so columns align:  "INFO    " vs "WARNING "
        level_padded = record.levelname.ljust(8)

        formatted = (
            f"{color}[{timestamp}] {level_padded}{self.RESET} "
            f"[{record.name}:{record.lineno}] "  # Module and line number
            f"{record.getMessage()}"
        )

        # Append exception traceback if present — readable multiline in terminal
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted



# Logger is primary entry point into python logging system and interface 
# that application code use directly to create log message 
# Logger factory = central function fro creating all logge rin the project
# All component(data pipeline,inference engine,API) call this 
# to get a consistenly configured logger

def get_logger(
    name: str,                          # Logger name, convention
    level: Optional[str]=None,          # Override log lvl (default: read from env)
    log_file: Optional[str]=None,       # Optional file path for file-based logging
    service_name: str = "ml_classifier",# Service identifier for log routing 
    environment: Optional[str]=None,    # Override environment (default: read from env)
        
) -> logging.Logger:
    """
    Create and return configured logger,
    Call once per module at module-lvl:
        logger = get_logger(__name__)
    Then use anwhere in the module:
        logger.info("Loaded model",extra = {"ml_context : {"version':"v3"}})
    """
    
    # Read configuration from environment variables.
    # WHY env vars? Lets ops teams change log verbosity without code changes or redeployment.
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")  # Default INFO in production
    log_format = os.getenv("LOG_FORMAT", "json")             # "json" or "console"
    env = environment or os.getenv("ENVIRONMENT", "production")

    # getLogger returns the SAME logger object if called with the same name.
    # This means multiple calls to get_logger(__name__) in the same module are safe.
    logger = logging.getLogger(name)

    # Guard: if this logger already has handlers, it was already configured.
    # Without this check, calling get_logger twice adds duplicate handlers → duplicate logs.
    if logger.handlers:
        return logger

    # Convert string level ("INFO") to logging constant (20).
    # getLevelName is bidirectional: it handles "INFO" → 20 and 20 → "INFO".
    numeric_level = logging.getLevelName(log_level_str.upper())
    if not isinstance(numeric_level, int):
        # Fallback if someone sets LOG_LEVEL=VERBOSE (not a real level)
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    # Prevent log records from bubbling up to the root logger.
    # WHY: The root logger may have a plain-text handler configured elsewhere;
    # propagating would cause duplicate or misformatted logs.
    logger.propagate = False

    # ---- Choose formatter based on LOG_FORMAT env var ----
    if log_format.lower() == "json":
        formatter = JSONFormatter(service_name=service_name, environment=env)
    else:
        # "console" or any other value → human-readable colored output for dev
        formatter = ColoredConsoleFormatter()

    # ---- STDOUT Handler (always present) ----
    # In containers (Docker/K8s), stdout is collected by the container runtime
    # and forwarded to your log aggregation system (Fluent Bit, Logstash, etc.)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # ---- Optional File Handler ----
    # Useful for local debugging or when running on bare VMs without a log shipper.
    # RotatingFileHandler prevents a single log file from consuming all disk space:
    #   maxBytes=10MB → at 10MB, rotate to .log.1
    #   backupCount=5 → keep .log.1 through .log.5 (50MB total max)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create parent dirs if needed
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # Keep 5 rotated files
            encoding="utf-8",           # Explicit encoding avoids locale-dependent issues
        )
        file_handler.setFormatter(formatter)  # Same formatter as stdout for consistency
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# ML-SPECIFIC LOGGING HELPERS
# These functions wrap the standard logger with domain-specific structure.
# Using them consistently means every prediction, metric, or model event
# has the same shape — enabling dashboards and alerts with zero extra work.
# =============================================================================

class MLLogger:
    """
    Opinionated wrapper around a standard Python logger.
    Provides methods that automatically inject ML-specific metadata
    (model name, run ID, stage) into every log entry.

    Usage:
        ml_logger = MLLogger(get_logger(__name__), model_name="churn_v2", run_id="abc-123")
        ml_logger.log_prediction(input_shape=(1, 45), predicted_class="churned", confidence=0.87)
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_name: str = "unknown",
        run_id: Optional[str] = None,   # Unique identifier for this prediction job/request
    ):
        self.logger = logger
        self.model_name = model_name
        # If no run_id is passed, generate one. This makes it possible to trace
        # all log entries for a single prediction request across multiple log lines.
        self.run_id = run_id or str(uuid.uuid4())

    def _base_context(self) -> dict:
        """
        Returns the metadata fields injected into every log call from this instance.
        Having model_name and run_id in every log line makes filtering trivial:
            SELECT * FROM logs WHERE ml_context.model_name = 'churn_v2'
        """
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
        }

    def log_prediction(
        self,
        input_shape: tuple,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        extra_context: Optional[dict] = None,
    ) -> None:
        """
        Logs a single completed prediction with its key metrics.
        Called after every successful model.predict() call.
        These logs feed real-time dashboards and SLA monitoring.
        """
        context = {
            **self._base_context(),
            "event": "prediction",          # Allows filtering only prediction events
            "input_shape": str(input_shape),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),  # Round to avoid float noise like 0.87000000001
            "latency_ms": round(latency_ms, 2),
            **(extra_context or {}),        # Merge any caller-provided extra fields
        }
        self.logger.info(
            f"Prediction: class={predicted_class}, confidence={confidence:.4f}, latency={latency_ms:.1f}ms",
            extra={"ml_context": context}
        )

    def log_model_loaded(self, model_path: str, load_time_ms: float) -> None:
        """
        Logs a successful model artifact load.
        WHY: Knowing when models are loaded (and how long it takes) is critical
        for cold start analysis in serverless or autoscaling environments.
        """
        context = {
            **self._base_context(),
            "event": "model_loaded",
            "model_path": model_path,
            "load_time_ms": round(load_time_ms, 2),
        }
        self.logger.info(
            f"Model loaded: {self.model_name} from {model_path} in {load_time_ms:.1f}ms",
            extra={"ml_context": context}
        )

    def log_data_validation(
        self,
        n_samples: int,
        n_features: int,
        validation_passed: bool,
        issues: Optional[list] = None,
    ) -> None:
        """
        Logs the result of input data validation.
        Tracking validation pass/fail rates over time reveals upstream data quality trends.
        """
        context = {
            **self._base_context(),
            "event": "data_validation",
            "n_samples": n_samples,
            "n_features": n_features,
            "validation_passed": validation_passed,
            "issues": issues or [],
        }
        level = logging.INFO if validation_passed else logging.WARNING
        self.logger.log(
            level,
            f"Data validation {'PASSED' if validation_passed else 'FAILED'}: "
            f"{n_samples} samples, {n_features} features",
            extra={"ml_context": context}
        )

    def log_batch_summary(
        self,
        batch_size: int,
        success_count: int,
        failure_count: int,
        total_latency_ms: float,
    ) -> None:
        """
        Logs a summary at the end of a batch prediction job.
        Provides a single log line per batch for job-level monitoring.
        """
        context = {
            **self._base_context(),
            "event": "batch_summary",
            "batch_size": batch_size,
            "success_count": success_count,
            "failure_count": failure_count,
            "failure_rate": round(failure_count / batch_size, 4) if batch_size > 0 else 0,
            "total_latency_ms": round(total_latency_ms, 2),
            "avg_latency_ms": round(total_latency_ms / batch_size, 2) if batch_size > 0 else 0,
        }
        level = logging.ERROR if failure_count > 0 else logging.INFO
        self.logger.log(
            level,
            f"Batch complete: {success_count}/{batch_size} succeeded in {total_latency_ms:.0f}ms",
            extra={"ml_context": context}
        )

    def log_exception(self, exc: Exception, stage: str, extra_context: Optional[dict] = None) -> None:
        """
        Logs a caught exception with full ML context.
        Use inside except blocks:
            try:
                ...
            except Exception as e:
                ml_logger.log_exception(e, stage="inference")
                raise
        """
        context = {
            **self._base_context(),
            "event": "exception",
            "stage": stage,
            "exception_type": type(exc).__name__,
            # If it's our custom exception, extract the structured dict
            "exception_detail": exc.to_dict() if hasattr(exc, "to_dict") else str(exc),
            **(extra_context or {}),
        }
        # exc_info=True appends the current exception's traceback to the log record
        self.logger.error(
            f"Exception in stage={stage}: {type(exc).__name__}: {exc}",
            exc_info=True,
            extra={"ml_context": context}
        )
