import os
import boto3
import argparse
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import tempfile

load_dotenv(dotenv_path=".env")

def get_latest_job_from_csv(bucket, prefix, s3_client):
    version_key = f"{prefix}/model_version.csv"

    # Download CSV from S3 to a temporary file
    with tempfile.NamedTemporaryFile(mode='r+b') as tmp:
        s3_client.download_file(bucket, version_key, tmp.name)
        df = pd.read_csv(tmp.name)

    if df.empty or 'job_name' not in df.columns:
        raise ValueError(f"❌ CSV file {version_key} is empty or missing 'job_name' column.")

    latest_row = df.sort_values('timestamp', ascending=False).iloc[0]
    job_name = latest_row['job_name']
    print(f"✅ Latest job from model_version.csv: {job_name}")
    return job_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True, help="Model config name (used as prefix)")
    parser.add_argument('--instance_type', type=str, default='ml.c5.4xlarge', help="Instance type for deployment")
    args = parser.parse_args()

    # Load environment variables
    REGION = os.environ['AWS_REGION']
    IAM_ROLE_ARN = os.environ['IAM_ROLE_NAME']
    PREFIX = args.model_config.replace("_", "-")

    # Set up session and clients
    boto_session = boto3.session.Session(region_name=REGION)
    sagemaker_session = Session(boto_session=boto_session)
    s3_client = boto3.client('s3', region_name=REGION)
    bucket = sagemaker_session.default_bucket()

    job_name = get_latest_job_from_csv(bucket, PREFIX, s3_client)

    model_data_s3_path = f"s3://{bucket}/{PREFIX}/output/{job_name}/output/model.tar.gz"
    print(f"✅ Model artifact path: {model_data_s3_path}")

    # === Create SKLearnModel ===
    sklearn_model = SKLearnModel(
        model_data=model_data_s3_path,
        role=IAM_ROLE_ARN,
        entry_point='inference.py',
        source_dir='src',
        framework_version='1.0-1',
        sagemaker_session=sagemaker_session,
        py_version='py3'
    )

    endpoint_name = f"{PREFIX}-endpoint"

    # === Deploy to endpoint ===
    print(f"🚀 Deploying model to endpoint: {endpoint_name} ...")
    predictor = sklearn_model.deploy(
        instance_type=args.instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )

    print(f"✅ Model deployed at endpoint: {endpoint_name}")

# aws cloudformation deploy \
#   --template-file src/CloudFormation.yml \
#   --stack-name ml \
#   --capabilities CAPABILITY_NAMED_IAM \
#   --parameter-overrides EndpointName=logistic-regression-endpoint

if __name__ == "__main__":
    main()
