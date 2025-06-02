import os
import boto3
import argparse
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import tempfile
import json

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

def deploy_single_model(model_config, instance_type, sagemaker_session, s3_client, iam_role_arn):
    """Deploy a single model"""
    PREFIX = model_config.replace("_", "-")
    bucket = sagemaker_session.default_bucket()
    
    job_name = get_latest_job_from_csv(bucket, PREFIX, s3_client)
    model_data_s3_path = f"s3://{bucket}/{PREFIX}/output/{job_name}/output/model.tar.gz"
    print(f"✅ Model artifact path for {model_config}: {model_data_s3_path}")
    
    # === Create SKLearnModel ===
    sklearn_model = SKLearnModel(
        model_data=model_data_s3_path,
        role=iam_role_arn,
        entry_point='inference.py',
        source_dir='src',
        framework_version='1.0-1',
        sagemaker_session=sagemaker_session,
        py_version='py3'
    )
    
    endpoint_name = f"{PREFIX}-endpoint"
    
    # === Deploy to endpoint ===
    print(f"🚀 Deploying model {model_config} to endpoint: {endpoint_name} ...")
    predictor = sklearn_model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )
    
    print(f"✅ Model {model_config} deployed at endpoint: {endpoint_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_configs', type=str, required=True, 
                       help="Comma-separated list of model config names (e.g., 'model1,model2,model3')")
    parser.add_argument('--instance_type', type=str, default='ml.m5.large', 
                       help="Instance type for deployment")
    args = parser.parse_args()
    
    # Load environment variables
    REGION = os.environ['AWS_REGION']
    IAM_ROLE_ARN = os.environ['IAM_ROLE_NAME']
    
    # Set up session and clients
    boto_session = boto3.session.Session(region_name=REGION)
    sagemaker_session = Session(boto_session=boto_session)
    s3_client = boto3.client('s3', region_name=REGION)
    
    # Parse model configs
    model_configs = [config.strip() for config in args.model_configs.split(',')]
    
    # Deploy all models
    deployed_models = []
    
    for model_config in model_configs:
        try:
            deploy_single_model(
                model_config, 
                args.instance_type, 
                sagemaker_session, 
                s3_client, 
                IAM_ROLE_ARN
            )
            deployed_models.append(model_config)
        except Exception as e:
            print(f"❌ Failed to deploy model {model_config}: {str(e)}")
            continue
    
    if not deployed_models:
        print("❌ No models were deployed successfully!")
        return
    
    print(f"\n✅ Successfully deployed {len(deployed_models)} models:")
    for model_config in deployed_models:
        endpoint_name = f"{model_config.replace('_', '-')}-endpoint"
        print(f"  - {model_config}: {endpoint_name}")
    
    print(f"\n✅ All models are ready!")
    print(f"💡 You can now deploy the unified API with CloudFormation")

if __name__ == "__main__":
    main()

# logitic_regression, mpl
# knn decision_tree naive_bayes random_forest
