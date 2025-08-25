# scripts/submit_training.py
import os
from datetime import datetime

import boto3

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn

# 1) Env (from ~/mlops-env.sh)
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
ROLE = os.environ["SM_ROLE_ARN"]
LABP = os.environ.get("LAB_PREFIX", "student")
S3_DATA_PROCESSED = os.environ["S3_DATA_PROCESSED"]  # .../data/processed
S3_ART_PREPROCESS = os.environ["S3_ART_PREPROCESS"]  # .../artifacts/preprocess
S3_ARTIFACTS = os.environ["S3_ARTIFACTS"]  # .../artifacts

# 2) Session
boto_sess = boto3.Session(region_name=REGION)
sm_sess = sagemaker.Session(boto_session=boto_sess)

# 3) Estimator
est = SKLearn(
    entry_point="sagemaker/code/train.py",
    role=ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sm_sess,
    base_job_name=f"{LABP}-train",
    # Ensure artifacts go to YOUR bucket/prefix
    output_path=f"{S3_ARTIFACTS}/training/",
    code_location=f"{S3_ARTIFACTS}/code/",
    hyperparameters={
        "target": "Churn",
        "max-iter": 200,
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class-weights": "auto",
        "random-state": 42,
    },
)

# 4) Channels (processed splits + preprocess artifacts)
inputs = {
    "train": TrainingInput(f"{S3_DATA_PROCESSED}/train/"),
    "val": TrainingInput(f"{S3_DATA_PROCESSED}/val/"),
    "test": TrainingInput(f"{S3_DATA_PROCESSED}/test/"),
    "artifacts": TrainingInput(f"{S3_ART_PREPROCESS}/artifacts/"),
}

# 5) Launch
job_name = f"{LABP}-train-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
print("Submitting job:", job_name)
est.fit(inputs=inputs, job_name=job_name, wait=True, logs="All")
print("Training job finished.")
