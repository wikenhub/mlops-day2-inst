# sagemaker/code/register.py
"""
Register a trained model + evaluation metrics into SageMaker Model Registry.

Assumptions:
  - Env vars set via mlops-env.sh: AWS_REGION, BUCKET, LAB_PREFIX, SM_ROLE_ARN, S3_ARTIFACTS
  - evaluation.json exists at s3://<BUCKET>/artifacts/evaluation/<job_name>/evaluation.json
  - model.tar.gz exists as output of the training job named <job_name>

Creates:
  - A new Model Package version inside the Model Package Group <LAB_PREFIX>-telco-churn
  - Status starts as PendingManualApproval (you can approve later)
"""

import json
import os

import boto3
import botocore

from sagemaker import image_uris

# ---- Env ----
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ["BUCKET"]
LABP = os.environ.get("LAB_PREFIX", "student")
SM_ROLE = os.environ["SM_ROLE_ARN"]
S3_ARTIFACTS = os.environ["S3_ARTIFACTS"]  # e.g., s3://.../artifacts

# ---- Clients ----
boto_sess = boto3.Session(region_name=REGION)
sm = boto_sess.client("sagemaker")
s3 = boto_sess.client("s3")


def latest_evaluation_key():
    """
    Return (job_name, key) for the most recent artifacts/evaluation/<job_name>/evaluation.json.
    Looks under both canonical and LAB_PREFIX-prefixed paths, with pagination.
    """
    prefixes = [
        "artifacts/evaluation/",
        f"{LABP}/artifacts/evaluation/",  # tolerate alternate layout if present
    ]
    candidates = []
    paginator = s3.get_paginator("list_objects_v2")
    for pref in prefixes:
        for page in paginator.paginate(Bucket=BUCKET, Prefix=pref):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith("/evaluation.json"):
                    candidates.append((obj["LastModified"], k))
    if not candidates:
        raise SystemExit("No evaluation.json found under artifacts/evaluation/. Run Lab 6 first.")
    last_mod, key = max(candidates, key=lambda x: x[0])
    job_name = key.rstrip("/").split("/")[-2]
    return job_name, key


def load_json_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def ensure_model_package_group(name: str) -> str:
    """
    Create (if needed) and return the Model Package Group ARN.
    """
    try:
        out = sm.describe_model_package_group(ModelPackageGroupName=name)
        return out["ModelPackageGroupArn"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise
        out = sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription="Telco churn model family",
            # Tags belong on the group (NOT individual versions); add here if desired.
            # Tags=[{"Key":"lab","Value":"lab7"},{"Key":"owner","Value":LABP}],
        )
        return out["ModelPackageGroupArn"]


def main():
    # 1) Find newest evaluation.json & parse it
    job_name, eval_key = latest_evaluation_key()
    evaluation = load_json_from_s3(BUCKET, eval_key)

    # 2) Find the training job's model artifact (model.tar.gz)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    model_data_url = desc["ModelArtifacts"]["S3ModelArtifacts"]

    # 3) Pull headline metrics for description
    pr_auc = float(evaluation["test"]["pr_auc"])
    roc_auc = float(evaluation["test"]["roc_auc"])
    # threshold_star may exist, but isn't required here

    # 4) Resolve framework inference image for region
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.2-1",
        image_scope="inference",
        py_version="py3",
    )

    # 5) Ensure group exists
    mpg_name = f"{LABP}-telco-churn"
    ensure_model_package_group(mpg_name)

    # 6) Point metrics to the eval JSON written by Lab 6
    eval_json_s3 = f"{S3_ARTIFACTS.rstrip('/')}/evaluation/{job_name}/evaluation.json"

    # 7) Create model package VERSION (no explicit name here)
    create_resp = sm.create_model_package(
        ModelPackageGroupName=mpg_name,
        ModelPackageDescription=(
            f"Telco churn LogReg. ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}. Job={job_name}"
        ),
        InferenceSpecification={
            "Containers": [{"Image": sklearn_image, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": eval_json_s3,
                }
            }
        },
        # NOTE: Do NOT set Tags here — tags are not supported on versions.
    )

    print("✓ Registered model:", create_resp["ModelPackageArn"])


if __name__ == "__main__":
    main()
