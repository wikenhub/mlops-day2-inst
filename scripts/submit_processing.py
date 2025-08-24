# scripts/submit_processing.py
import os
from datetime import datetime

import boto3

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# --- 1) Read environment (set in earlier labs) ---
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ["BUCKET"]  # e.g., stu01-<acct>-ap-northeast-2-mlops
S3_DATA = os.environ["S3_DATA"]  # e.g., s3://.../data
S3_ART = os.environ["S3_ARTIFACTS"]  # e.g., s3://.../artifacts
ROLE = os.environ["SM_ROLE_ARN"]  # your per-student SageMaker role
LABP = os.environ.get("LAB_PREFIX", "stuXX")

# Optional: these control split behavior for the script
TARGET = os.environ.get("TARGET_COL", "Churn")
TEST_P = float(os.environ.get("TEST_SIZE", "0.20"))
VAL_P = float(os.environ.get("VAL_SIZE", "0.10"))
SEED = int(os.environ.get("RANDOM_STATE", "42"))

# --- 2) Set up a SageMaker session bound to your region ---
boto_sess = boto3.Session(region_name=REGION)
sm_sess = sagemaker.Session(boto_session=boto_sess)

# --- 3) Choose a container and instance type for Processing ---
# We use SKLearnProcessor because our script imports scikit-learn/pandas/numpy.
# If you ever hit an image-version error in your region, try a nearby version (e.g., "1.3-1").
processor = SKLearnProcessor(
    framework_version="1.2-1",  # SageMaker-provided sklearn image (includes numpy/pandas)
    role=ROLE,
    instance_type="ml.m5.large",  # balanced CPU for preprocessing; change if needed
    instance_count=1,
    base_job_name=f"{LABP}-preprocess",
    sagemaker_session=sm_sess,
)

# --- 4) Wire S3 <-> container paths ---
# S3 (left) will be mounted inside the container (right).
# Our script defaults to these /opt/ml/processing/* locations.
inputs = [
    ProcessingInput(
        source=f"{S3_DATA}/raw/telco/",
        destination="/opt/ml/processing/input",
    ),
]

outputs = [
    ProcessingOutput(  # train split CSV
        output_name="train",
        source="/opt/ml/processing/train",
        destination=f"{S3_DATA}/processed/train/",
    ),
    ProcessingOutput(  # val split CSV
        output_name="val",
        source="/opt/ml/processing/val",
        destination=f"{S3_DATA}/processed/val/",
    ),
    ProcessingOutput(  # test split CSV
        output_name="test",
        source="/opt/ml/processing/test",
        destination=f"{S3_DATA}/processed/test/",
    ),
    ProcessingOutput(  # fitted transformers, metadata
        output_name="artifacts",
        source="/opt/ml/processing/artifacts",
        destination=f"{S3_ART}/preprocess/",
    ),
    ProcessingOutput(  # human-readable EDA report
        output_name="report",
        source="/opt/ml/processing/report",
        destination=f"{S3_ART}/preprocess/",
    ),
]

# --- 5) Kick off the job ---
job_name = f"{LABP}-preprocess-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

# Optional: attach the job to an Experiment/Trial for grouping
# (you can comment this out if not using Experiments yet)
experiment_config = {
    "ExperimentName": f"{LABP}-churn-exp",
    "TrialName": f"{LABP}-preprocess",
    "TrialComponentDisplayName": "preprocess",
}

print(f"Submitting Processing job: {job_name}")
processor.run(
    job_name=job_name,
    code="sagemaker/code/preprocess.py",  # path inside your repo
    inputs=inputs,
    outputs=outputs,
    arguments=[
        "--target",
        TARGET,
        "--test-size",
        str(TEST_P),
        "--val-size",
        str(VAL_P),
        "--random-state",
        str(SEED),
    ],
    wait=True,  # stream logs until the job finishes
    logs=True,  # show CloudWatch logs in your terminal
    experiment_config=experiment_config,
)
print("Processing job finished.")
