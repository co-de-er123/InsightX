import boto3
import time
from typing import Dict, List, Optional
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EMRManager:
    """
    A class to manage AWS EMR clusters and job submissions.
    Handles cluster creation, job submission, and monitoring.
    """
    
    def __init__(self, region_name: str = 'us-west-2', log_uri: str = 's3://emr-logs/'):
        """
        Initialize the EMR manager with AWS credentials and configuration.
        
        Args:
            region_name (str): AWS region name
            log_uri (str): S3 URI for EMR logs
        """
        self.region_name = region_name
        self.log_uri = log_uri
        self.emr_client = boto3.client('emr', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
    
    def create_cluster(
        self,
        cluster_name: str,
        instance_type: str = 'm5.xlarge',
        instance_count: int = 3,
        release_label: str = 'emr-6.8.0',
        applications: List[str] = None,
        ec2_key_name: Optional[str] = None,
        subnet_id: Optional[str] = None,
        security_group_ids: Optional[List[str]] = None,
        job_flow_role: str = 'EMR_EC2_DefaultRole',
        service_role: str = 'EMR_DefaultRole',
        visible_to_all_users: bool = True,
        tags: Optional[Dict[str, str]] = None,
        keep_alive: bool = False
    ) -> str:
        """
        Create an EMR cluster.
        
        Args:
            cluster_name (str): Name of the cluster
            instance_type (str): EC2 instance type
            instance_count (int): Number of instances
            release_label (str): EMR release label
            applications (List[str]): List of applications to install
            ec2_key_name (str): EC2 key pair name for SSH access
            subnet_id (str): VPC subnet ID
            security_group_ids (List[str]): List of security group IDs
            job_flow_role (str): IAM role for EC2 instances
            service_role (str): IAM role for EMR service
            visible_to_all_users (bool): Whether the cluster is visible to all IAM users
            tags (Dict[str, str]): Tags to apply to the cluster
            keep_alive (bool): Whether to keep the cluster alive after steps complete
            
        Returns:
            str: Cluster ID
        """
        if applications is None:
            applications = ['Spark', 'Hadoop', 'Hive', 'Livy']
        
        if tags is None:
            tags = {
                'Project': 'TelecomChurnPrediction',
                'Environment': 'Production'
            }
        
        try:
            response = self.emr_client.run_job_flow(
                Name=cluster_name,
                LogUri=self.log_uri,
                ReleaseLabel=release_label,
                Applications=[{'Name': app} for app in applications],
                Instances={
                    'InstanceGroups': [
                        {
                            'Name': 'Master node',
                            'Market': 'ON_DEMAND',
                            'InstanceRole': 'MASTER',
                            'InstanceType': instance_type,
                            'InstanceCount': 1,
                        },
                        {
                            'Name': 'Worker nodes',
                            'Market': 'ON_DEMAND',
                            'InstanceRole': 'CORE',
                            'InstanceType': instance_type,
                            'InstanceCount': instance_count - 1,
                        }
                    ],
                    'Ec2KeyName': ec2_key_name,
                    'KeepJobFlowAliveWhenNoSteps': keep_alive,
                    'TerminationProtected': False,
                    'Ec2SubnetId': subnet_id,
                },
                Steps=[],  # Steps will be added later
                JobFlowRole=job_flow_role,
                ServiceRole=service_role,
                VisibleToAllUsers=visible_to_all_users,
                Tags=[{'Key': k, 'Value': v} for k, v in tags.items()],
                SecurityConfiguration='EMR_DefaultSecurityConfiguration',
                ScaleDownBehavior='TERMINATE_AT_TASK_COMPLETION',
                Configurations=[
                    {
                        'Classification': 'spark',
                        'Properties': {
                            'maximizeResourceAllocation': 'true'
                        }
                    },
                    {
                        'Classification': 'spark-defaults',
                        'Properties': {
                            'spark.dynamicAllocation.enabled': 'true',
                            'spark.shuffle.service.enabled': 'true',
                            'spark.sql.shuffle.partitions': '200',
                            'spark.default.parallelism': '200'
                        }
                    }
                ]
            )
            
            cluster_id = response['JobFlowId']
            logger.info(f"Created EMR cluster {cluster_id}")
            return cluster_id
            
        except ClientError as e:
            logger.error(f"Failed to create EMR cluster: {e}")
            raise
    
    def add_step(
        self,
        cluster_id: str,
        name: str,
        script_path: str,
        script_args: List[str] = None,
        action_on_failure: str = 'CONTINUE'
    ) -> str:
        """
        Add a step to an existing EMR cluster.
        
        Args:
            cluster_id (str): ID of the EMR cluster
            name (str): Name of the step
            script_path (str): S3 path to the script
            script_args (List[str]): Arguments to pass to the script
            action_on_failure (str): Action to take if the step fails
            
        Returns:
            str: Step ID
        """
        if script_args is None:
            script_args = []
            
        try:
            response = self.emr_client.add_job_flow_steps(
                JobFlowId=cluster_id,
                Steps=[
                    {
                        'Name': name,
                        'ActionOnFailure': action_on_failure,
                        'HadoopJarStep': {
                            'Jar': 'command-runner.jar',
                            'Args': [
                                'spark-submit',
                                '--deploy-mode', 'cluster',
                                '--master', 'yarn',
                                '--conf', 'spark.yarn.submit.waitAppCompletion=true',
                                script_path
                            ] + script_args
                        }
                    }
                ]
            )
            
            step_id = response['StepIds'][0]
            logger.info(f"Added step {step_id} to cluster {cluster_id}")
            return step_id
            
        except ClientError as e:
            logger.error(f"Failed to add step to EMR cluster: {e}")
            raise
    
    def submit_spark_job(
        self,
        cluster_id: str,
        name: str,
        script_path: str,
        script_args: List[str] = None,
        action_on_failure: str = 'CONTINUE',
        wait_for_completion: bool = True,
        poll_interval: int = 30
    ) -> Dict:
        """
        Submit a Spark job to an EMR cluster and optionally wait for completion.
        
        Args:
            cluster_id (str): ID of the EMR cluster
            name (str): Name of the job
            script_path (str): S3 path to the script
            script_args (List[str]): Arguments to pass to the script
            action_on_failure (str): Action to take if the job fails
            wait_for_completion (bool): Whether to wait for job completion
            poll_interval (int): Interval in seconds to poll job status
            
        Returns:
            Dict: Job status information
        """
        step_id = self.add_step(
            cluster_id=cluster_id,
            name=name,
            script_path=script_path,
            script_args=script_args,
            action_on_failure=action_on_failure
        )
        
        if wait_for_completion:
            return self.wait_for_step(cluster_id, step_id, poll_interval)
        
        return {'StepId': step_id, 'Status': 'PENDING'}
    
    def wait_for_step(
        self,
        cluster_id: str,
        step_id: str,
        poll_interval: int = 30
    ) -> Dict:
        """
        Wait for a step to complete.
        
        Args:
            cluster_id (str): ID of the EMR cluster
            step_id (str): ID of the step to wait for
            poll_interval (int): Interval in seconds to poll step status
            
        Returns:
            Dict: Step status information
        """
        logger.info(f"Waiting for step {step_id} to complete...")
        
        while True:
            response = self.emr_client.describe_step(
                ClusterId=cluster_id,
                StepId=step_id
            )
            
            status = response['Step']['Status']
            state = status['State']
            
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                logger.info(f"Step {step_id} {state}: {status.get('StateChangeReason', '')}")
                return {
                    'StepId': step_id,
                    'State': state,
                    'Status': status,
                    'Timeline': response['Step'].get('Timeline', {})
                }
            
            logger.info(f"Step {step_id} status: {state}")
            time.sleep(poll_interval)
    
    def terminate_cluster(self, cluster_id: str) -> None:
        """
        Terminate an EMR cluster.
        
        Args:
            cluster_id (str): ID of the cluster to terminate
        """
        try:
            self.emr_client.terminate_job_flows(JobFlowIds=[cluster_id])
            logger.info(f"Terminated EMR cluster {cluster_id}")
        except ClientError as e:
            logger.error(f"Failed to terminate EMR cluster {cluster_id}: {e}")
            raise
    
    def upload_file_to_s3(
        self,
        local_path: str,
        bucket: str,
        s3_key: str
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path (str): Local path to the file
            bucket (str): S3 bucket name
            s3_key (str): S3 object key
            
        Returns:
            str: S3 URI of the uploaded file
        """
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
            s3_uri = f"s3://{bucket}/{s3_key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise

def submit_etl_job(
    cluster_id: str,
    input_path: str,
    output_path: str,
    region: str = 'us-west-2',
    script_bucket: str = 'telecom-churn-scripts',
    script_key: str = 'scripts/etl_job.py'
) -> Dict:
    """
    Submit an ETL job to an EMR cluster.
    
    Args:
        cluster_id (str): ID of the EMR cluster
        input_path (str): S3 path to input data
        output_path (str): S3 path for output data
        region (str): AWS region
        script_bucket (str): S3 bucket containing the ETL script
        script_key (str): S3 key of the ETL script
        
    Returns:
        Dict: Job status information
    """
    emr = EMRManager(region_name=region)
    
    # Submit the ETL job
    job_status = emr.submit_spark_job(
        cluster_id=cluster_id,
        name="Telecom Churn ETL",
        script_path=f"s3://{script_bucket}/{script_key}",
        script_args=[input_path, output_path]
    )
    
    return job_status

def submit_training_job(
    cluster_id: str,
    data_path: str,
    model_path: str,
    region: str = 'us-west-2',
    script_bucket: str = 'telecom-churn-scripts',
    script_key: str = 'scripts/train_model.py'
) -> Dict:
    """
    Submit a model training job to an EMR cluster.
    
    Args:
        cluster_id (str): ID of the EMR cluster
        data_path (str): S3 path to training data
        model_path (str): S3 path to save the trained model
        region (str): AWS region
        script_bucket (str): S3 bucket containing the training script
        script_key (str): S3 key of the training script
        
    Returns:
        Dict: Job status information
    """
    emr = EMRManager(region_name=region)
    
    # Submit the training job
    job_status = emr.submit_spark_job(
        cluster_id=cluster_id,
        name="Telecom Churn Model Training",
        script_path=f"s3://{script_bucket}/{script_key}",
        script_args=[data_path, model_path]
    )
    
    return job_status
