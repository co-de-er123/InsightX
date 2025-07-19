from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import logging
import mlflow
import mlflow.spark
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Gradient Boosted Trees model for telecom churn prediction.
    Handles model training, evaluation, and persistence.
    """
    
    def __init__(self):
        self.spark = self._create_spark_session()
        self.model = None
        self.feature_columns = [
            'account_length', 'international_plan', 'voice_mail_plan',
            'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
            'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
            'total_eve_charge', 'total_night_minutes', 'total_night_calls',
            'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
            'total_intl_charge', 'customer_service_calls', 'total_minutes',
            'total_calls', 'total_charge', 'avg_minutes_per_call', 'daytime_ratio'
        ]
    
    def _create_spark_session(self):
        """Initialize and return a Spark session with MLlib configuration."""
        return SparkSession.builder \
            .appName("ChurnPrediction") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
    
    def load_data(self, data_path):
        """
        Load processed data from parquet files.
        
        Args:
            data_path (str): Path to the processed parquet data
            
        Returns:
            DataFrame: Loaded Spark DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        return self.spark.read.parquet(data_path)
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        
        Args:
            df (DataFrame): Input DataFrame with raw features
            
        Returns:
            DataFrame: DataFrame with feature vector
        """
        logger.info("Preparing features...")
        
        # Assemble features into a vector
        assembler = VectorAssembler(
            inputCols=self.feature_columns,
            outputCol="features"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # Create pipeline for feature preparation
        pipeline = Pipeline(stages=[assembler, scaler])
        
        # Fit and transform the data
        model = pipeline.fit(df)
        df_prepared = model.transform(df)
        
        return df_prepared, model
    
    def train_model(self, train_df, val_df=None):
        """
        Train a Gradient Boosted Trees model.
        
        Args:
            train_df (DataFrame): Training data
            val_df (DataFrame, optional): Validation data for early stopping
            
        Returns:
            GBTModel: Trained model
        """
        logger.info("Training Gradient Boosted Trees model...")
        
        # Initialize GBT Classifier
        gbt = GBTClassifier(
            labelCol="churn",
            featuresCol="scaled_features",
            maxIter=100,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
            featureSubsetStrategy="sqrt",
            maxBins=32,
            minInstancesPerNode=20,
            minInfoGain=0.01,
            seed=42
        )
        
        # Create parameter grid for hyperparameter tuning
        param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [4, 5, 6]) \
            .addGrid(gbt.stepSize, [0.05, 0.1, 0.2]) \
            .build()
        
        # Define evaluator (F1 score for imbalanced data)
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="churn", 
            metricName="f1"
        )
        
        # Set up cross-validation
        cv = CrossValidator(
            estimator=gbt,
            estimatorParamMaps=param_grid,
            evaluator=f1_evaluator,
            numFolds=3,
            seed=42,
            parallelism=2
        )
        
        # Train the model
        cv_model = cv.fit(train_df)
        
        # Get the best model
        self.model = cv_model.bestModel
        
        logger.info(f"Best model parameters: {self.model.extractParamMap()}")
        
        return self.model
    
    def evaluate_model(self, model, test_df):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Initialize evaluators
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="churn", 
            metricName="f1"
        )
        
        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="churn",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        # Calculate metrics
        f1 = f1_evaluator.evaluate(predictions)
        auc = auc_evaluator.evaluate(predictions)
        
        # Get feature importances
        feature_importances = list(zip(self.feature_columns, 
                                     model.featureImportances.toArray()))
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        metrics = {
            "f1_score": f1,
            "auc_roc": auc,
            "feature_importances": dict(feature_importances[:10])  # Top 10 features
        }
        
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info("Top 10 important features:")
        for feature, importance in feature_importances[:10]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        return metrics
    
    def save_model(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        logger.info(f"Saving model to {model_path}")
        self.model.save(model_path)
    
    def log_to_mlflow(self, params, metrics, model_path, experiment_name="telecom_churn"):
        """
        Log model and metrics to MLflow.
        
        Args:
            params (dict): Model parameters
            metrics (dict): Evaluation metrics
            model_path (str): Path to save the model
            experiment_name (str): MLflow experiment name
        """
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics({
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics["auc_roc"]
            })
            
            # Log model
            mlflow.spark.log_model(
                self.model,
                "model",
                registered_model_name="telecom_churn_gbt"
            )
            
            # Log feature importance plot
            self._log_feature_importance_plot(metrics["feature_importances"])
            
            # Log the model path
            mlflow.log_artifact(model_path)
    
    def _log_feature_importance_plot(self, feature_importances):
        """Create and log feature importance plot."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create a DataFrame for plotting
            df_importance = pd.DataFrame(
                list(feature_importances.items()),
                columns=['feature', 'importance']
            )
            
            # Create plot
            plt.figure(figsize=(10, 6))
            df_importance.sort_values('importance').plot(
                kind='barh', 
                x='feature', 
                y='importance',
                legend=False
            )
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save and log the plot
            plot_path = "feature_importance.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {str(e)}")

def main():
    # Example usage
    data_path = "data/processed/telecom_processed"
    model_path = "models/telecom_churn_model"
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load and prepare data
    df = predictor.load_data(data_path)
    df_prepared, feature_pipeline = predictor.prepare_features(df)
    
    # Split data
    train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    model = predictor.train_model(train_df)
    
    # Evaluate model
    metrics = predictor.evaluate_model(model, test_df)
    
    # Save model
    predictor.save_model(model_path)
    
    # Log to MLflow
    params = {
        "model_type": "Gradient Boosted Trees",
        "max_depth": model.getMaxDepth(),
        "max_iter": model.getMaxIter(),
        "step_size": model.getStepSize(),
        "feature_columns": predictor.feature_columns
    }
    
    predictor.log_to_mlflow(params, metrics, model_path)
    
    logger.info("Model training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
