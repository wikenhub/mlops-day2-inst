# sagemaker/code/register.py
"""
Register a trained model + metrics into SageMaker Model Registry.

Inputs:
  * LAB_PREFIX, BUCKET, S3_ARTIFACTS, SM_ROLE_ARN from env
  * evaluation.json in s3://.../artifacts/evaluation/<job_name>/
  * model.tar.gz in training output (from Lab 5/6)

Outputs:
  * New Model Package version under the Model Package Group
  * Pending or Approved status, with metrics attached
"""

import json
import os

import boto3
import botocore

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ["BUCKET"]
LABP = os.environ.get("LAB_PREFIX", "student")
SM_ROLE = os.environ["SM_ROLE_ARN"]
S3_ARTIFACTS = os.environ["S3_ARTIFACTS"]

boto_sess = boto3.Session(region_name=REGION)
sm = boto_sess.client("sagemaker")
s3 = boto_sess.client("s3")


def latest_evaluation():
    """Return (job_name, evaluation dict, s3_key) of most recent evaluation.json."""
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="artifacts/evaluation/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/evaluation.json"):
                candidates.append((obj["LastModified"], obj["Key"]))
    if not candidates:
        raise SystemExit("No evaluation.json found; run Lab 6 first.")
    last_mod, key = max(candidates, key=lambda x: x[0])
    job = key.rstrip("/").split("/")[-2]
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return job, json.loads(body), key


def ensure_group(name: str):
    try:
        return sm.describe_model_package_group(ModelPackageGroupName=name)["ModelPackageGroupArn"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise
        return sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription="Telco churn model family",
        )["ModelPackageGroupArn"]


def main():
    job, evaluation, key = latest_evaluation()
    desc = sm.describe_training_job(TrainingJobName=job)
    model_data_url = desc["ModelArtifacts"]["S3ModelArtifacts"]

    pr_auc = float(evaluation["test"]["pr_auc"])
    roc_auc = float(evaluation["test"]["roc_auc"])

    from sagemaker import image_uris

    sklearn_image = image_uris.retrieve("sklearn", REGION, "1.2-1", "inference", "py3")

    mpg = f"{LABP}-telco-churn"
    ensure_group(mpg)

    eval_json_s3 = f"{S3_ARTIFACTS.rstrip('/')}/evaluation/{job}/evaluation.json"

    create_resp = sm.create_model_package(
        ModelPackageGroupName=mpg,
        ModelPackageDescription=(
            f"Telco churn LogReg. ROC={roc_auc:.3f}, PR={pr_auc:.3f}. Job={job}"
        ),
        InferenceSpecification={
            "Containers": [{"Image": sklearn_image, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": eval_json_s3}
            }
        },
    )

    print("✓ Registered model:", create_resp["ModelPackageArn"])


if __name__ == "__main__":
    main()
