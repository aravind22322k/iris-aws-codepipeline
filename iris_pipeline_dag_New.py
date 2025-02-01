m_
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import json
import subprocess

# AWS configurations
S3_BUCKET = "iris-airflow-bucket"
MODEL_ARTIFACT = "iris_model.tar.gz"
ROLE_ARN = "arn:aws:iam::605134428434:role/SageMakerRole"  # Replace with your SageMaker role ARN
REGION = "us-east-1"

def prepare_data():
    # In this example, we assume the Iris dataset is already in S3.
    print("Data preparation skipped. Dataset already exists in S3.")

def train_model():
    client = boto3.client("sagemaker", region_name=REGION)

    training_job_name = "iris-training-job"
    training_image = "605134428434.dkr.ecr.us-east-1.amazonaws.com/ecr-iris-repo1:latest"
    input_data_s3 = f"s3://iris-airflow-bucket/iris_train.csv/"
    output_data_s3 = f"s3://iris-airflow-bucket/iris-output/"

    response = client.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_data_s3,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        OutputDataConfig={"S3OutputPath": output_data_s3},
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )
    print(f"Training job started: {response['TrainingJobArn']}")

def evaluate_model():
    print("Evaluating model... Assuming evaluation passed.")
    # Implement actual evaluation logic here
    return True

def deploy_model():
    client = boto3.client("sagemaker", region_name=REGION)

    model_name = "iris-model"
    endpoint_config_name = "iris-endpoint-config"
    endpoint_name = "iris-endpoint"

    model_data = f"s3://iris-airflow-bucket/iris-output/iris_model.tar.gz"
    container = {
       "Image": "605134428434.dkr.ecr.us-east-1.amazonaws.com/ecr-iris-repo1:latest",
       "ModelDataUrl": model_data
     }

    client.create_model(ModelName=model_name, PrimaryContainer=container, ExecutionRoleArn=ROLE_ARN)
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "c6gd.metal",
            }
        ],
    )
    client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    print(f"Model deployed at endpoint: {endpoint_name}")

def test_endpoint():
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    endpoint_name = "iris-endpoint"
    payload = json.dumps({"instances": [[5.1, 3.5, 1.4, 0.2]]})

    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=payload
    )
    print(f"Prediction result: {response['Body'].read().decode()}")

# Define the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "iris_pipeline",
    default_args=default_args,
    description="Iris Model Pipeline with Airflow and SageMaker",
    schedule_interval=None,  # Trigger manually
    start_date=datetime(2024, 12, 15),
    catchup=False,
) as dag:

    prepare_task = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    test_task = PythonOperator(task_id="test_endpoint", python_callable=test_endpoint)

    prepare_task >> train_task >> evaluate_task >> deploy_task >> test_task