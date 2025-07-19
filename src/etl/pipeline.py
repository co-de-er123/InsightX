from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnull, mean, stddev
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelecomETLPipeline:
    """
    ETL Pipeline for processing telecom churn data.
    Handles data loading, cleaning, and feature engineering.
    """
    
    def __init__(self):
        self.spark = self._create_spark_session()
        
    def _create_spark_session(self):
        """Initialize and return a Spark session."""
        return SparkSession.builder \
            .appName("TelecomChurnETL") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
    
    def load_data(self, file_path):
        """
        Load telecom data from CSV file.
        
        Args:
            file_path (str): Path to the input CSV file
            
        Returns:
            DataFrame: Loaded Spark DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        # Define schema for better type control
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("account_length", IntegerType(), True),
            StructField("area_code", StringType(), True),
            StructField("international_plan", StringType(), True),
            StructField("voice_mail_plan", StringType(), True),
            StructField("number_vmail_messages", IntegerType(), True),
            StructField("total_day_minutes", DoubleType(), True),
            StructField("total_day_calls", IntegerType(), True),
            StructField("total_day_charge", DoubleType(), True),
            StructField("total_eve_minutes", DoubleType(), True),
            StructField("total_eve_calls", IntegerType(), True),
            StructField("total_eve_charge", DoubleType(), True),
            StructField("total_night_minutes", DoubleType(), True),
            StructField("total_night_calls", IntegerType(), True),
            StructField("total_night_charge", DoubleType(), True),
            StructField("total_intl_minutes", DoubleType(), True),
            StructField("total_intl_calls", IntegerType(), True),
            StructField("total_intl_charge", DoubleType(), True),
            StructField("customer_service_calls", IntegerType(), True),
            StructField("churn", StringType(), True)
        ])
        
        return self.spark.read.csv(file_path, header=True, schema=schema, inferSchema=False)
    
    def clean_data(self, df):
        """
        Clean the telecom data by handling missing values and outliers.
        
        Args:
            df (DataFrame): Input Spark DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Convert 'yes'/'no' to binary (1/0)
        df = df.withColumn("churn", when(col("churn") == "yes", 1).otherwise(0))
        df = df.withColumn("international_plan", when(col("international_plan") == "yes", 1).otherwise(0))
        df = df.withColumn("voice_mail_plan", when(col("voice_mail_plan") == "yes", 1).otherwise(0))
        
        # Handle missing values
        numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType))]
        
        for col_name in numeric_cols:
            # Calculate mean and standard deviation for each numeric column
            stats = df.select(mean(col_name).alias('mean'), stddev(col_name).alias('std')).collect()[0]
            mean_val = stats['mean']
            std_val = stats['std']
            
            # Cap outliers at 3 standard deviations
            upper_bound = mean_val + 3 * std_val
            lower_bound = mean_val - 3 * std_val
            
            df = df.withColumn(
                col_name,
                when(col(col_name).isNull(), mean_val)
                .when(col(col_name) > upper_bound, upper_bound)
                .when(col(col_name) < lower_bound, lower_bound)
                .otherwise(col(col_name))
            )
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features from existing data.
        
        Args:
            df (DataFrame): Input Spark DataFrame
            
        Returns:
            DataFrame: DataFrame with new features
        """
        logger.info("Performing feature engineering...")
        
        # Calculate total minutes, calls, and charges
        df = df.withColumn("total_minutes", 
                          col("total_day_minutes") + 
                          col("total_eve_minutes") + 
                          col("total_night_minutes"))
        
        df = df.withColumn("total_calls", 
                          col("total_day_calls") + 
                          col("total_eve_calls") + 
                          col("total_night_calls"))
        
        df = df.withColumn("total_charge", 
                          col("total_day_charge") + 
                          col("total_eve_charge") + 
                          col("total_night_charge") + 
                          col("total_intl_charge"))
        
        # Calculate average minutes per call
        df = df.withColumn("avg_minutes_per_call", 
                          when(col("total_calls") > 0, 
                               col("total_minutes") / col("total_calls"))
                          .otherwise(0))
        
        # Calculate call duration ratios
        df = df.withColumn("daytime_ratio", 
                          col("total_day_minutes") / col("total_minutes"))
        
        return df
    
    def run_pipeline(self, input_path, output_path):
        """
        Run the complete ETL pipeline.
        
        Args:
            input_path (str): Path to input data
            output_path (str): Path to save processed data
            
        Returns:
            DataFrame: Processed DataFrame
        """
        logger.info("Starting ETL pipeline...")
        start_time = datetime.now()
        
        try:
            # Load data
            df = self.load_data(input_path)
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Feature engineering
            df_features = self.feature_engineering(df_clean)
            
            # Save processed data
            df_features.write.parquet(output_path, mode="overwrite")
            
            # Log completion
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"ETL pipeline completed in {duration:.2f} seconds")
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error in ETL pipeline: {str(e)}")
            raise

def main():
    # Example usage
    input_path = "data/raw/telecom_data.csv"
    output_path = "data/processed/telecom_processed"
    
    pipeline = TelecomETLPipeline()
    df = pipeline.run_pipeline(input_path, output_path)
    
    # Show sample data
    df.show(5)

if __name__ == "__main__":
    main()
