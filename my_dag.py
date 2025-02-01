from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator, SageMakerEndpointOperator
from airflow.utils.dates import days_ago
import boto3

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

# Define the DAG
with DAG(
    dag_id='iris_pipeline',
    default_args=default_args,
    description='A pipeline for Iris prediction using SageMaker',
    schedule_interval=None,  # Set to a cron expression if you want it scheduled
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # Task 1: Upload dataset to S3
    def upload_dataset_to_s3():
        s3 = S3Hook()
        bucket_name = 'iris-airflow-bucket'
        key = 'iris.csv'
        local_file = 'iris.csv'  # Path to the local dataset file
        s3.load_file(filename=local_file, bucket_name=bucket_name, key=key, replace=True)

    upload_dataset_task = PythonOperator(
        task_id='upload_dataset_to_s3',
        python_callable=upload_dataset_to_s3,
    )

    # Task 2: Train the SageMaker model
    sagemaker_training_config = {
        'TrainingJobName': 'iris-training-job',
        'AlgorithmSpecification': {
            'TrainingImage': '382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
            'TrainingInputMode': 'File'
        },
        'RoleArn': 'arn:aws:iam::123456789012:role/SageMakerRole',  # Replace with your SageMaker IAM Role ARN
        'InputDataConfig': [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://iris-airflow-bucket/iris-data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': 's3://iris-airflow-bucket/iris-output/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600
        }
    }

    train_model_task = SageMakerTrainingOperator(
        task_id='train_model',
        config=sagemaker_training_config,
        aws_conn_id='aws_default',  # Airflow connection ID for AWS
        wait_for_completion=True
    )

    # Task 3: Deploy the trained model
    sagemaker_endpoint_config = {
        'EndpointConfigName': 'iris-endpoint-config',
        'ProductionVariants': [
            {
                'VariantName': 'AllTraffic',
                'ModelName': 'iris-model',  # Ensure this matches the model created during training
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 1
            }
        ]
    }

    deploy_model_task = SageMakerEndpointOperator(
        task_id='deploy_model',
        config={
            'ModelName': 'iris-model',  # Ensure this matches the trained model
            'EndpointName': 'iris-endpoint',
            'EndpointConfig': sagemaker_endpoint_config
        },
        aws_conn_id='aws_default',
        wait_for_completion=True
    )

    # Define the pipeline sequence
    upload_dataset_task >> train_model_task >> deploy_model_task
