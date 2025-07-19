"""
Configuration settings for the Telecom Churn Prediction project.

This file contains all the configuration parameters used throughout the project,
including Spark, Kafka, AWS EMR, and model training settings.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
STREAMING_DATA_DIR = DATA_DIR / "streaming"

# Create data subdirectories
for subdir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, STREAMING_DATA_DIR]:
    subdir.mkdir(parents=True, exist_ok=True)

# Sample data file (for development)
SAMPLE_DATA_PATH = RAW_DATA_DIR / "telecom_data.csv"

# Spark configuration
class SparkConfig:
    """Spark configuration parameters."""
    
    # General Spark settings
    APP_NAME = "TelecomChurnPrediction"
    MASTER = "local[*]"  # Use "yarn" for cluster mode
    
    # Memory and cores
    EXECUTOR_MEMORY = "4g"
    DRIVER_MEMORY = "4g"
    EXECUTOR_CORES = 2
    
    # Parallelism and partitions
    DEFAULT_PARALLELISM = 8
    SHUFFLE_PARTITIONS = 8
    
    # Serialization
    SERIALIZER = "org.apache.spark.serializer.KryoSerializer"
    
    # Dynamic allocation
    DYNAMIC_ALLOCATION_ENABLED = "true"
    SHUFFLE_SERVICE_ENABLED = "true"
    
    # SQL settings
    SQL_SHUFFLE_PARTITIONS = 8
    SQL_AUTO_BROADCASTJOIN_THRESHOLD = "-1"  # Disable broadcast join
    
    @classmethod
    def get_config(cls) -> Dict[str, str]:
        """Get Spark configuration as a dictionary."""
        return {
            "spark.app.name": cls.APP_NAME,
            "spark.master": cls.MASTER,
            "spark.executor.memory": cls.EXECUTOR_MEMORY,
            "spark.driver.memory": cls.DRIVER_MEMORY,
            "spark.executor.cores": str(cls.EXECUTOR_CORES),
            "spark.default.parallelism": str(cls.DEFAULT_PARALLELISM),
            "spark.sql.shuffle.partitions": str(cls.SHUFFLE_PARTITIONS),
            "spark.serializer": cls.SERIALIZER,
            "spark.dynamicAllocation.enabled": cls.DYNAMIC_ALLOCATION_ENABLED,
            "spark.shuffle.service.enabled": cls.SHUFFLE_SERVICE_ENABLED,
            "spark.sql.autoBroadcastJoinThreshold": cls.SQL_AUTO_BROADCASTJOIN_THRESHOLD,
            # Add more Spark configurations as needed
        }

# Kafka configuration
class KafkaConfig:
    """Kafka configuration parameters."""
    
    # Kafka broker settings
    BOOTSTRAP_SERVERS = ["localhost:9092"]
    TOPIC = "telecom-usage"
    GROUP_ID = "telecom-churn-group"
    
    # Consumer settings
    AUTO_OFFSET_RESET = "latest"  # or 'earliest'
    ENABLE_AUTO_COMMIT = "false"
    MAX_POLL_RECORDS = 1000
    SESSION_TIMEOUT_MS = 30000
    HEARTBEAT_INTERVAL_MS = 10000
    
    # Producer settings
    ACKS = "all"
    RETRIES = 3
    BATCH_SIZE = 16384
    LINGER_MS = 1
    BUFFER_MEMORY = 33554432
    
    # Schema Registry (if using Confluent Schema Registry)
    SCHEMA_REGISTRY_URL = "http://localhost:8081"
    
    @classmethod
    def get_consumer_config(cls) -> Dict[str, str]:
        """Get Kafka consumer configuration."""
        return {
            "bootstrap.servers": ",".join(cls.BOOTSTRAP_SERVERS),
            "group.id": cls.GROUP_ID,
            "auto.offset.reset": cls.AUTO_OFFSET_RESET,
            "enable.auto.commit": cls.ENABLE_AUTO_COMMIT,
            "max.poll.records": str(cls.MAX_POLL_RECORDS),
            "session.timeout.ms": str(cls.SESSION_TIMEOUT_MS),
            "heartbeat.interval.ms": str(cls.HEARTBEAT_INTERVAL_MS)
        }
    
    @classmethod
    def get_producer_config(cls) -> Dict[str, str]:
        """Get Kafka producer configuration."""
        return {
            "bootstrap.servers": ",".join(cls.BOOTSTRAP_SERVERS),
            "acks": cls.ACKS,
            "retries": str(cls.RETRIES),
            "batch.size": str(cls.BATCH_SIZE),
            "linger.ms": str(cls.LINGER_MS),
            "buffer.memory": str(cls.BUFFER_MEMORY)
        }

# AWS EMR Configuration
class EMRConfig:
    """AWS EMR configuration parameters."""
    
    # Cluster configuration
    REGION = "us-west-2"
    RELEASE_LABEL = "emr-6.8.0"
    INSTANCE_TYPE = "m5.xlarge"
    INSTANCE_COUNT = 3
    EC2_KEY_NAME = "your-key-pair"  # Replace with your EC2 key pair
    SUBNET_ID = "subnet-xxxxxxxx"  # Replace with your subnet ID
    
    # IAM roles
    JOB_FLOW_ROLE = "EMR_EC2_DefaultRole"
    SERVICE_ROLE = "EMR_DefaultRole"
    
    # Applications to install
    APPLICATIONS = ["Spark", "Hadoop", "Hive", "Livy"]
    
    # S3 paths
    LOG_URI = "s3://emr-logs/"
    SCRIPTS_BUCKET = "telecom-churn-scripts"
    
    # EMR steps
    STEPS = [
        {
            "Name": "Setup Debugging",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["state-pusher-script"]
            }
        }
    ]
    
    @classmethod
    def get_cluster_config(cls) -> Dict[str, Any]:
        """Get EMR cluster configuration."""
        return {
            "Name": "Telecom-Churn-Prediction",
            "LogUri": cls.LOG_URI,
            "ReleaseLabel": cls.RELEASE_LABEL,
            "Applications": [{"Name": app} for app in cls.APPLICATIONS],
            "Instances": {
                "InstanceGroups": [
                    {
                        "Name": "Master node",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "MASTER",
                        "InstanceType": cls.INSTANCE_TYPE,
                        "InstanceCount": 1,
                    },
                    {
                        "Name": "Worker nodes",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "CORE",
                        "InstanceType": cls.INSTANCE_TYPE,
                        "InstanceCount": cls.INSTANCE_COUNT - 1,
                    }
                ],
                "Ec2KeyName": cls.EC2_KEY_NAME,
                "KeepJobFlowAliveWhenNoSteps": True,
                "TerminationProtected": False,
                "Ec2SubnetId": cls.SUBNET_ID,
            },
            "Steps": cls.STEPS,
            "JobFlowRole": cls.JOB_FLOW_ROLE,
            "ServiceRole": cls.SERVICE_ROLE,
            "VisibleToAllUsers": True,
            "Tags": [
                {"Key": "Project", "Value": "TelecomChurnPrediction"},
                {"Key": "Environment", "Value": "Production"}
            ],
            "Configurations": [
                {
                    "Classification": "spark",
                    "Properties": {
                        "maximizeResourceAllocation": "true"
                    }
                },
                {
                    "Classification": "spark-defaults",
                    "Properties": {
                        "spark.dynamicAllocation.enabled": "true",
                        "spark.shuffle.service.enabled": "true",
                        "spark.sql.shuffle.partitions": "200",
                        "spark.default.parallelism": "200"
                    }
                }
            ]
        }

# Model training configuration
class ModelConfig:
    """Model training and evaluation configuration."""
    
    # Data split
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    VALIDATION_RATIO = 0.0  # If using cross-validation
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Feature columns (will be set dynamically)
    FEATURE_COLUMNS = [
        'account_length', 'international_plan', 'voice_mail_plan',
        'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
        'total_eve_charge', 'total_night_minutes', 'total_night_calls',
        'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
        'total_intl_charge', 'customer_service_calls', 'total_minutes',
        'total_calls', 'total_charge', 'avg_minutes_per_call', 'daytime_ratio'
    ]
    
    # Target column
    TARGET_COLUMN = 'churn'
    
    # Gradient Boosted Trees parameters
    GBT_PARAMS = {
        'maxDepth': 5,
        'maxBins': 32,
        'minInstancesPerNode': 10,
        'minInfoGain': 0.0,
        'maxMemoryInMB': 256,
        'cacheNodeIds': True,
        'checkpointInterval': 10,
        'lossType': 'logistic',
        'maxIter': 100,
        'stepSize': 0.1,
        'seed': RANDOM_SEED,
        'subsamplingRate': 0.8,
        'featureSubsetStrategy': 'sqrt'
    }
    
    # Cross-validation parameters
    CV_FOLDS = 3
    CV_PARALLELISM = 2
    
    # Hyperparameter grid for tuning
    PARAM_GRID = {
        'maxDepth': [3, 5, 7],
        'maxBins': [16, 32, 64],
        'minInstancesPerNode': [5, 10, 20],
        'stepSize': [0.05, 0.1, 0.2]
    }
    
    # Evaluation metrics
    EVAL_METRICS = ['f1', 'areaUnderROC', 'accuracy', 'weightedPrecision', 'weightedRecall']
    
    # Model persistence
    MODEL_SAVE_PATH = str(MODEL_DIR / "telecom_churn_model")
    MODEL_VERSION = "1.0.0"

# MLflow configuration
class MLflowConfig:
    """MLflow configuration for experiment tracking."""
    
    TRACKING_URI = "http://localhost:5000"
    EXPERIMENT_NAME = "telecom_churn_prediction"
    
    # Enable automatic logging of parameters, metrics, and models
    AUTOLOG = True
    
    # Log model artifacts
    LOG_MODEL = True
    
    # Log batch metrics
    LOG_BATCH_METRICS = True
    
    @classmethod
    def setup_mlflow(cls):
        """Set up MLflow tracking."""
        import mlflow
        
        # Set the tracking URI
        mlflow.set_tracking_uri(cls.TRACKING_URI)
        
        # Set the experiment
        mlflow.set_experiment(cls.EXPERIMENT_NAME)
        
        # Enable autologging if specified
        if cls.AUTOLOG:
            mlflow.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=cls.LOG_MODEL,
                log_batch_metrics=cls.LOG_BATCH_METRICS
            )

# Logging configuration
class LoggingConfig:
    """Logging configuration."""
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = str(LOG_DIR / "telecom_churn.log")
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration."""
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(cls.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter(cls.LOG_FORMAT)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Set log level for specific loggers
        logging.getLogger('py4j').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.INFO)
        logging.getLogger('s3transfer').setLevel(logging.INFO)
        logging.getLogger('urllib3').setLevel(logging.INFO)

# Initialize logging when the module is imported
LoggingConfig.setup_logging()

# Example usage
if __name__ == "__main__":
    # Example of how to use the configuration
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Spark app name: {SparkConfig.APP_NAME}")
    print(f"Kafka topic: {KafkaConfig.TOPIC}")
    print(f"EMR region: {EMRConfig.REGION}")
    print(f"Model save path: {ModelConfig.MODEL_SAVE_PATH}")
