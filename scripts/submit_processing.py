# scripts/submit_processing.py
import os
from datetime import datetime

import boto3

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# --- 1) Read environment (set in earlier labs) ---
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ["BUCKET"]  # e.g., stuxx-<acct>-ap-northeast-2-mlops
ROLE = os.environ["SM_ROLE_ARN"]  # your per-student SageMaker role
LABP = os.environ.get("LAB_PREFIX", "stuXX")

S3_CODE = os.environ["S3_CODE"]
S3_DATA_RAW = os.environ["S3_DATA_RAW"]
S3_DATA_PROCESSED = os.environ["S3_DATA_PROCESSED"]
S3_ART_PREPROCESS = os.environ["S3_ART_PREPROCESS"]

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
        source=f"{os.environ['S3_DATA_RAW']}/telco/",
        destination="/opt/ml/processing/input",
    ),
]

outputs = [
    ProcessingOutput(
        output_name="train",
        source="/opt/ml/processing/train",
        destination=f"{S3_DATA_PROCESSED}/train/",
    ),
    ProcessingOutput(
        output_name="val",
        source="/opt/ml/processing/val",
        destination=f"{S3_DATA_PROCESSED}/val/",
    ),
    ProcessingOutput(
        output_name="test",
        source="/opt/ml/processing/test",
        destination=f"{S3_DATA_PROCESSED}/test/",
    ),
    ProcessingOutput(  # fitted transformers + schema/columns
        output_name="artifacts",
        source="/opt/ml/processing/artifacts",
        destination=f"{S3_ART_PREPROCESS}/artifacts/",
    ),
    ProcessingOutput(  # EDA report
        output_name="report",
        source="/opt/ml/processing/report",
        destination=f"{S3_ART_PREPROCESS}/report/",
    ),
]

# --- 5) Kick off the job ---
job_name = f"{LABP}-preprocess-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

print(f"Submitting Processing job: {job_name}")
processor.run(
    job_name=job_name,
    code=f"{S3_CODE.rstrip('/')}/preprocess.py",  # <-- S3 URI to the script
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
    wait=True,
    logs=True,
)
print("Processing job finished.")
