import argparse
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os
from pathlib import Path
import tarfile
import tempfile
import pandas as pd
import re
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=".env")

def upload_to_s3(local_path, s3_key, bucket_name, s3_client):
    if not Path(local_path).exists():
        raise FileNotFoundError(f"⚠️ File {local_path} không tồn tại.")
    s3_client.upload_file(local_path, bucket_name, s3_key)
    s3_uri = f's3://{bucket_name}/{s3_key}'
    print(f"✅ Uploaded {local_path} → {s3_uri}")
    return s3_uri

def append_model_version_to_s3(job_name, metrics, bucket_name, key):
    s3 = boto3.client('s3')
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(bucket_name, key, tmp.name)
            df = pd.read_csv(tmp.name)
    except Exception as e:
        print(f"⚠️ Failed to download or read file: {e}")
        df = pd.DataFrame(columns=["timestamp", "job_name", "accuracy", "precision", "recall", "f1_score"])

    new_row = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "job_name": job_name,
        "accuracy": metrics.get("Accuracy"),
        "precision": metrics.get("Precision"),
        "recall": metrics.get("Recall"),
        "f1_score": metrics.get("F1 Score")
    }])

    df = pd.concat([df, new_row], ignore_index=True)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmp_csv:
        df.to_csv(tmp_csv.name, index=False)
        s3.upload_file(tmp_csv.name, bucket_name, key)
    print(f"✅ Appended model version info to s3://{bucket_name}/{key}")

def run_single_job(model_config, session, s3_client, bucket_name, region, role_arn, delete_files=False):
    prefix = model_config.replace("_", "-")

    # === Upload dữ liệu ===
    train_data_s3_path = upload_to_s3("./data/train.csv", f"{prefix}/data/train/train.csv", bucket_name, s3_client)
    val_data_s3_path = upload_to_s3("./data/val.csv", f"{prefix}/data/val/val.csv", bucket_name, s3_client)

    # === Khởi tạo SageMaker job ===
    sklearn = SKLearn(
        entry_point='train.py',
        source_dir='src',
        role=role_arn,
        instance_type='ml.c5.4xlarge',
        framework_version='1.0-1',
        py_version='py3',
        output_path=f's3://{bucket_name}/{prefix}/output',
        code_location=f's3://{bucket_name}/{prefix}/code',
        hyperparameters={
            'config': f'/opt/ml/code/model_config/{model_config}.yaml',
            'train_path': '/opt/ml/input/data/train/train.csv',
            'val_path': '/opt/ml/input/data/val/val.csv',
            'output_dir': '/opt/ml/model'
        }
    )

    sklearn.fit({
        'train': f's3://{bucket_name}/{prefix}/data/train/',
        'val': f's3://{bucket_name}/{prefix}/data/val/'
    })

    # === Trích xuất metrics và log chi tiết ===
    job_name = sklearn.latest_training_job.name
    print(f"✅ SageMaker job name: {job_name}")

    local_tar_path = f"model_{model_config}.tar.gz"
    model_s3_key = f"{prefix}/output/{job_name}/output/model.tar.gz"
    s3_client.download_file(bucket_name, model_s3_key, local_tar_path)

    metrics = {}
    detail_log_text = ""

    with tarfile.open(local_tar_path, "r:gz") as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            log_path = Path(tmpdir) / "training_log.txt"
            if log_path.exists():
                with open(log_path, "r") as f:
                    lines = f.readlines()

                start_index = next((i for i, line in enumerate(lines) if "Validation results" in line), None)
                if start_index is not None:
                    selected_lines = lines[start_index:]
                    detail_log_text = "".join(selected_lines)

                    for line in selected_lines:
                        for key in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                            match = re.search(rf"{key}:\s*([0-9.]+)", line)
                            if match:
                                metrics[key] = float(match.group(1))

                    print(f"✅ Đã trích xuất log chi tiết cho {model_config}")
                else:
                    print("⚠️ 'Validation results' không tìm thấy trong training_log.txt")
            else:
                print("⚠️ training_log.txt không tồn tại trong model artifact.")

    append_model_version_to_s3(
        job_name=job_name,
        metrics=metrics,
        bucket_name=bucket_name,
        key=f"{prefix}/model_version.csv"
    )

    if delete_files:
        for file in [local_tar_path]:
            try:
                os.remove(file)
                print(f"🗑️ Đã xóa file {file}")
            except FileNotFoundError:
                print(f"⚠️ File {file} không tồn tại để xóa")

    return detail_log_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, nargs='+', required=True,
                        help="Danh sách tên file YAML trong thư mục model_config (không có phần mở rộng .yaml)")
    parser.add_argument('--delete', action='store_true',
                        help="Nếu có cờ này, sẽ xóa file model sau khi train")
    args = parser.parse_args()

    REGION = os.environ['AWS_REGION']
    IAM_ROLE_ARN = os.environ['IAM_ROLE_NAME']

    session = sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION))
    BUCKET_NAME = session.default_bucket()
    s3_client = boto3.client('s3', region_name=REGION)

    all_details = []
    for model_config in args.model_config:
        print(f"\n=== 🚀 Bắt đầu train với config: {model_config} ===")
        try:
            detail_log = run_single_job(
                model_config=model_config,
                session=session,
                s3_client=s3_client,
                bucket_name=BUCKET_NAME,
                region=REGION,
                role_arn=IAM_ROLE_ARN,
                delete_files=args.delete
            )
            if detail_log:
                header = f"\n\n===== Log chi tiết cho model: {model_config} =====\n"
                all_details.append(header + detail_log)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý config '{model_config}': {e}")

    if all_details:
        with open("detail.txt", "w") as f:
            f.writelines(all_details)
        print("✅ Đã ghi toàn bộ log chi tiết vào detail.txt")

if __name__ == "__main__":
    main()
