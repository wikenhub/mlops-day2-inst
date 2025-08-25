# scripts/submit_training.py
import argparse
import os
from datetime import datetime

import boto3

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn

# 0) Helper function to parse the job submit parameters.
# These parameters are parsed when we submit the job from our DevBox.
# We wrap it in SKLearn estimator and pass it to SageMaker


def parse_submit_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Churn")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("-C", "--C", type=float, default=1.0)
    p.add_argument("--penalty", default="l2", choices=["l2"])
    p.add_argument("--solver", default="lbfgs", choices=["lbfgs", "liblinear", "saga"])
    p.add_argument("--class-weights", default="auto", choices=["auto", "balanced", "none"])
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


args = parse_submit_args()

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
        "target": args.target,
        "max-iter": args.max_iter,
        "C": args.C,
        "penalty": args.penalty,
        "solver": args.solver,
        "class-weights": args.class_weights,
        "random-state": args.random_state,
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
