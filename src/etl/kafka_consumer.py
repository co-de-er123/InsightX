from kafka import KafkaConsumer, TopicPartition
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import json
import logging
from typing import Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaStreamProcessor:
    """
    A class to process real-time telecom data from Kafka using Spark Structured Streaming.
    Handles data consumption, processing, and writing to a sink.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'telecom-usage',
        checkpoint_location: str = 'checkpoints/telecom_stream',
        output_path: str = 'data/streaming/telecom_processed',
        processing_interval: str = '60 seconds'
    ):
        """
        Initialize the Kafka stream processor.
        
        Args:
            bootstrap_servers (str): Kafka bootstrap servers
            topic (str): Kafka topic to consume from
            checkpoint_location (str): Path for Spark checkpointing
            output_path (str): Path to write processed data
            processing_interval (str): Processing interval for micro-batches
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.checkpoint_location = checkpoint_location
        self.output_path = output_path
        self.processing_interval = processing_interval
        
        # Initialize Spark session
        self.spark = self._create_spark_session()
        
        # Define schema for incoming JSON data
        self.schema = self._define_schema()
    
    def _create_spark_session(self):
        """Create and return a Spark session with Kafka support."""
        return SparkSession.builder \
            .appName("TelecomKafkaStream") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location) \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
            .getOrCreate()
    
    @staticmethod
    def _define_schema() -> StructType:
        """Define the schema for incoming Kafka messages."""
        return StructType([
            StructField("customer_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("call_duration", DoubleType(), True),
            StructField("call_type", StringType(), True),  # 'local', 'international', 'roaming'
            StructField("call_direction", StringType(), True),  # 'incoming', 'outgoing'
            StructField("call_successful", IntegerType(), True),  # 0 or 1
            StructField("data_usage_mb", DoubleType(), True),
            StructField("sms_count", IntegerType(), True),
            StructField("roaming", IntegerType(), True),  # 0 or 1
            StructField("device_type", StringType(), True),
            StructField("network_type", StringType(), True),
            StructField("location_lat", DoubleType(), True),
            StructField("location_lon", DoubleType(), True)
        ])
    
    def read_from_kafka(self):
        """
        Read data from Kafka topic using Spark Structured Streaming.
        
        Returns:
            DataFrame: Streaming DataFrame with parsed Kafka messages
        """
        logger.info(f"Reading from Kafka topic: {self.topic}")
        
        return self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load() \
            .selectExpr("CAST(value AS STRING) as json") \
            .select(from_json("json", self.schema).alias("data")) \
            .select("data.*")
    
    def process_stream(self, df):
        """
        Process the streaming DataFrame.
        
        Args:
            df (DataFrame): Input streaming DataFrame
            
        Returns:
            DataFrame: Processed streaming DataFrame
        """
        logger.info("Processing streaming data...")
        
        # Add processing time column
        processed_df = df.withColumn("processing_time", col("timestamp"))
        
        # Add derived columns
        processed_df = processed_df.withColumn(
            "is_roaming", 
            (col("roaming") == 1).cast("integer")
        )
        
        return processed_df
    
    def write_to_sink(self, df, output_mode: str = "append"):
        """
        Write the streaming DataFrame to a sink.
        
        Args:
            df (DataFrame): Streaming DataFrame to write
            output_mode (str): Output mode (append, update, complete)
            
        Returns:
            StreamingQuery: The streaming query object
        """
        logger.info(f"Writing to sink: {self.output_path}")
        
        return df.writeStream \
            .outputMode(output_mode) \
            .format("parquet") \
            .option("path", self.output_path) \
            .option("checkpointLocation", self.checkpoint_location) \
            .trigger(processingTime=self.processing_interval) \
            .start()
    
    def start_stream(self):
        """Start the streaming application."""
        logger.info("Starting Kafka stream processing...")
        
        try:
            # Read from Kafka
            kafka_df = self.read_from_kafka()
            
            # Process the stream
            processed_df = self.process_stream(kafka_df)
            
            # Write to sink
            query = self.write_to_sink(processed_df)
            
            # Wait for the stream to terminate
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            raise

class KafkaMessageProducer:
    """
    A helper class to produce sample messages to Kafka for testing.
    """
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'telecom-usage'):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers (str): Kafka bootstrap servers
            topic (str): Kafka topic to produce to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
    
    def connect(self):
        """Connect to the Kafka broker."""
        from kafka import KafkaProducer
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise
    
    def generate_sample_message(self) -> Dict[str, Any]:
        """
        Generate a sample telecom usage message.
        
        Returns:
            Dict: Sample message data
        """
        import random
        from datetime import datetime, timedelta
        
        call_types = ['local', 'international', 'roaming']
        directions = ['incoming', 'outgoing']
        device_types = ['smartphone', 'tablet', 'feature_phone', 'iot']
        network_types = ['4G', '5G', 'LTE', '3G']
        
        return {
            "customer_id": f"cust_{random.randint(10000, 99999)}",
            "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(0, 60))).isoformat() + "Z",
            "call_duration": round(random.uniform(0.1, 30.0), 2),
            "call_type": random.choice(call_types),
            "call_direction": random.choice(directions),
            "call_successful": random.choices([0, 1], weights=[0.05, 0.95])[0],
            "data_usage_mb": round(random.uniform(0.1, 50.0), 2),
            "sms_count": random.randint(0, 10),
            "roaming": 1 if random.random() < 0.1 else 0,
            "device_type": random.choice(device_types),
            "network_type": random.choice(network_types),
            "location_lat": round(37.7749 + random.uniform(-0.5, 0.5), 4),
            "location_lon": round(-122.4194 + random.uniform(-0.5, 0.5), 4)
        }
    
    def send_message(self, message: Dict[str, Any] = None):
        """
        Send a message to the Kafka topic.
        
        Args:
            message (Dict): Message to send. If None, generates a sample message.
        """
        if message is None:
            message = self.generate_sample_message()
        
        try:
            self.producer.send(self.topic, value=message)
            self.producer.flush()
            logger.debug(f"Sent message: {json.dumps(message, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise
    
    def generate_test_data(self, num_messages: int = 100, delay: float = 0.1):
        """
        Generate test data and send to Kafka.
        
        Args:
            num_messages (int): Number of messages to generate
            delay (float): Delay between messages in seconds
        """
        logger.info(f"Generating {num_messages} test messages...")
        
        for i in range(num_messages):
            self.send_message()
            time.sleep(delay)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Sent {i + 1}/{num_messages} messages")
        
        logger.info("Test data generation complete")

def start_stream_processing():
    """Start the Kafka stream processing application."""
    processor = KafkaStreamProcessor(
        bootstrap_servers='localhost:9092',
        topic='telecom-usage',
        checkpoint_location='checkpoints/telecom_stream',
        output_path='data/streaming/telecom_processed',
        processing_interval='60 seconds'
    )
    
    try:
        processor.start_stream()
    except KeyboardInterrupt:
        logger.info("Stopping stream processing...")
    except Exception as e:
        logger.error(f"Error in stream processing: {str(e)}")

def generate_test_data():
    """Generate test data and send to Kafka."""
    producer = KafkaMessageProducer(
        bootstrap_servers='localhost:9092',
        topic='telecom-usage'
    )
    
    try:
        producer.connect()
        producer.generate_test_data(num_messages=1000, delay=0.5)
    except KeyboardInterrupt:
        logger.info("Stopping test data generation...")
    except Exception as e:
        logger.error(f"Error generating test data: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Stream Processor for Telecom Data')
    parser.add_argument('--mode', type=str, choices=['process', 'generate'], 
                       default='process',
                       help='Mode: process (consume and process) or generate (produce test data)')
    
    args = parser.parse_args()
    
    if args.mode == 'process':
        start_stream_processing()
    else:
        generate_test_data()
