# scripts/submit_batch_transform.py
"""
Run a SageMaker Batch Transform using the latest Approved model package in <LAB_PREFIX>-telco-churn.
Mirrors the working notebook flow (describe package → SKLearnModel → transformer), but:
  - uploads source_dir.tar.gz to YOUR project bucket via code_location
  - passes FEATURE_LIST_JSON and EVAL_JSON_S3 to inference.py
  - avoids dereferencing fields that can be None (no 'subscriptable' crashes)

Env expected (exported by ~/mlops-env.sh):
  AWS_REGION, LAB_PREFIX, BUCKET, S3_ARTIFACTS, SM_ROLE_ARN
"""

import logging
import os
import uuid
from datetime import datetime

import boto3

from sagemaker import Session
from sagemaker.sklearn.model import SKLearnModel

for name in ("sagemaker", "boto3", "botocore", "urllib3", "s3transfer"):
    logging.getLogger(name).setLevel(logging.WARNING)


# --- Env / clients ---
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ["BUCKET"]
LABP = os.environ.get("LAB_PREFIX", "student")
SM_ROLE_ARN = os.environ["SM_ROLE_ARN"]
S3_ARTIFACTS = os.environ["S3_ARTIFACTS"].rstrip("/")  # e.g., s3://<bucket>/<prefix>

boto_sess = boto3.Session(region_name=REGION)
sm = boto_sess.client("sagemaker")
s3 = boto_sess.client("s3")
sess = Session(boto_session=boto_sess)


def latest_approved_pkg_arn(group: str) -> str:
    r = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    lst = r.get("ModelPackageSummaryList", [])
    if not lst:
        raise SystemExit("No Approved model package. Approve one in Lab 7.")
    return lst[0]["ModelPackageArn"]


def describe_package_bits(pkg_arn: str):
    info = sm.describe_model_package(ModelPackageName=pkg_arn)
    c = info["InferenceSpecification"]["Containers"][0]
    image_uri = c["Image"]
    model_data_url = c["ModelDataUrl"]
    eval_json_s3 = info["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    return image_uri, model_data_url, eval_json_s3


def find_test_input() -> str:
    cands = []
    p = s3.get_paginator("list_objects_v2")
    for prefix in ("data/processed/", "artifacts/preprocess/"):
        for page in p.paginate(Bucket=BUCKET, Prefix=prefix):
            for o in page.get("Contents", []) or []:
                if o["Key"].endswith("test.csv"):
                    cands.append((o["LastModified"], o["Key"]))
    if not cands:
        raise SystemExit("No processed test.csv found.")
    _, key = max(cands, key=lambda x: x[0])
    return f"s3://{BUCKET}/{key}"


def main():
    group = f"{LABP}-telco-churn"
    pkg_arn = latest_approved_pkg_arn(group)
    image_uri, model_data_url, eval_json_s3 = describe_package_bits(pkg_arn)

    # Contract (columns) published by preprocessing
    feature_list_s3 = f"{S3_ARTIFACTS}/preprocess/columns.json"

    # Build a Model that packages our inference.py; keep code in OUR bucket.
    sk_model = SKLearnModel(
        model_data=model_data_url,
        image_uri=image_uri,
        role=SM_ROLE_ARN,
        entry_point="inference.py",
        source_dir="sagemaker/code",
        sagemaker_session=sess,
        code_location=f"{S3_ARTIFACTS}/code",  # <-- keeps source_dir.tar.gz in project bucket
        env={
            "EVAL_JSON_S3": eval_json_s3,
            "FEATURE_LIST_JSON": feature_list_s3,
        },
    )

    # Transformer (ephemeral batch job)
    out = f"{S3_ARTIFACTS}/batch/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}/"
    transformer = sk_model.transformer(
        instance_type="ml.m5.large",
        instance_count=1,
        strategy="MultiRecord",
        assemble_with="Line",
        output_path=out,
        max_payload=6,
        max_concurrent_transforms=2,
    )

    # Submit job (no log streaming)
    job_name = f"{LABP}-bt-{uuid.uuid4().hex[:8]}"
    input_s3 = find_test_input()

    # Start the transform but do NOT wait here (avoids streaming container logs)
    transformer.transform(
        data=input_s3,
        content_type="text/csv",
        split_type="Line",
        job_name=job_name,
        wait=False,  # <-- key: don't stream logs
    )

    print(f"Started: {job_name}")
    print(f"Input  : {input_s3}")
    print(f"Output : {out}")  # single source of truth for where predictions will land

    # Quiet poll loop (minimal status line; no container logs)
    import time

    while True:
        d = sm.describe_transform_job(TransformJobName=job_name)
        status = d.get("TransformJobStatus", "Unknown")
        print("Status:", status)
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                # print a concise failure reason if present (still quiet)
                fr = d.get("FailureReason")
                if fr:
                    print("FailureReason:", fr)
            # Optionally, confirm the exact S3 output SageMaker returns
            s3_path = (d.get("TransformOutput") or {}).get("S3OutputPath") or out
            print("Final output:", s3_path)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
