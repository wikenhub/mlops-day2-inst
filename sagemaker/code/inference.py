# sagemaker/code/inference.py
"""
Inference entry point for SageMaker (scikit-learn container).
Implements model_fn / input_fn / predict_fn / output_fn.

Contract:
- Input: CSV with header; drops 'Churn' label if present.
- Output: CSV with columns: proba,pred
"""

import io
import json
import os
from urllib.parse import urlparse

import boto3
import joblib
import numpy as np
import pandas as pd


# --- 1) Load the model from /opt/ml/model ---
def model_fn(model_dir: str):
    model = joblib.load(f"{model_dir}/model.joblib")
    # If training saved a dict, extract the estimator
    if isinstance(model, dict):
        for key in ("model", "estimator", "clf", "sk_model"):
            if key in model:
                model = model[key]
                break
    return model


# --- 2) Parse incoming request ---
# --- input_fn: read CSV, drop label, enforce feature contract ---


# --- imports used by both helpers and input_fn ---
def _read_json_anywhere(path_or_s3uri):
    """Read JSON from local path or s3://bucket/key and return the parsed object."""
    if path_or_s3uri.startswith("s3://"):
        u = urlparse(path_or_s3uri)
        bucket = u.netloc
        key = u.path.lstrip("/")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    else:
        with open(path_or_s3uri) as f:
            return json.load(f)


def _load_columns():
    """
    Return a list[str] of column names from columns.json.
    Accepts either:
      - {"columns": [...]}  (preferred)
      - [...]               (raw list)
    """
    # priority: explicit env override -> model directory default
    candidates = []
    if os.getenv("FEATURE_LIST_JSON"):
        candidates.append(os.getenv("FEATURE_LIST_JSON"))
    candidates.append("/opt/ml/model/columns.json")

    last_err = None
    for src in candidates:
        try:
            data = _read_json_anywhere(src)
            # normalize shape
            if isinstance(data, dict) and "columns" in data:
                cols = data["columns"]
            elif isinstance(data, list):
                cols = data
            else:
                raise ValueError(f"Unsupported columns.json structure from {src}: {type(data)}")

            # final validation
            if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
                raise ValueError(f"columns must be a list[str]; got {type(cols)} from {src}")

            # success
            print(f"[input_fn] Loaded {len(cols)} columns from {src}")
            return cols

        except Exception as e:
            last_err = e
            print(f"[input_fn] WARN: failed to load columns from {src}: {e}")

    raise RuntimeError(f"Could not load columns.json from any source. Last error: {last_err}")


def input_fn(input_data, content_type):
    """
    Parse incoming CSV using schema from columns.json.
    - Drops 'label' if present in schema.
    - Requires header row and all required feature columns.
    - Returns numpy array in the training feature order.
    """
    if content_type != "text/csv":
        raise ValueError(f"Unsupported content type: {content_type}")

    expected = _load_columns()
    feature_cols = [c for c in expected if c != "label"]

    # read CSV (expects header)
    df = pd.read_csv(io.StringIO(input_data))

    # quick visibility (first 5 cols)
    print(f"[input_fn] Incoming header count={len(df.columns)} first5={list(df.columns[:5])}")
    print(f"[input_fn] Expected features count={len(feature_cols)} first5={feature_cols[:5]}")

    # require all features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            "This violates the preprocessing→inference contract."
        )

    # select in the exact training order; keep any extra columns out
    X = df[feature_cols].to_numpy(dtype="float32")
    print(f"[input_fn] Inference matrix shape: {X.shape} (rows, features)")
    return X


# --- 3) Run prediction ---
# --- threshold discovery (at module import time) ---
# Default

THRESHOLD = 0.5  # fallback


def _parse_s3(uri: str):
    """Split s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri}")
    b, k = uri[5:].split("/", 1)
    return b, k


def _load_eval_from_s3(s3_client, s3_uri: str) -> dict:
    b, k = _parse_s3(s3_uri)
    obj = s3_client.get_object(Bucket=b, Key=k)
    return json.loads(obj["Body"].read())


def _discover_threshold() -> float:
    # 1) explicit override wins
    val = os.environ.get("THRESHOLD")
    if val:
        try:
            return float(val)
        except ValueError:
            print(f"[WARN] THRESHOLD not a float: {val!r}; ignoring")

    # 2) exact evaluation.json S3 location
    eval_uri = os.environ.get("EVAL_JSON_S3")
    if eval_uri:
        try:
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
            evaluation = _load_eval_from_s3(s3, eval_uri)
            return float(evaluation["test"]["threshold_star"])
        except Exception as e:
            print(f"[WARN] Could not read EVAL_JSON_S3 ({eval_uri}): {e}")

    # 3) fallback
    print("[INFO] Using default threshold=0.5 (no THRESHOLD or EVAL_JSON_S3).")
    return 0.5


THRESHOLD = _discover_threshold()
print(f"[INFO] Decision threshold in use: {THRESHOLD:.3f}")


def predict_fn(parsed_input, model):
    if isinstance(model, dict):
        # defensive in case model_fn didn’t unwrap
        for key in ("model", "estimator", "clf", "sk_model"):
            if key in model:
                model = model[key]
                break
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Loaded model of type {type(model)} lacks predict_proba")
    proba_pos = model.predict_proba(parsed_input)[:, 1]
    pred = (proba_pos >= THRESHOLD).astype(int)
    return np.column_stack([proba_pos, pred])


# --- 4) Serialize output ---
def output_fn(prediction, accept: str):
    """
    Serialize to CSV with header 'proba,pred'.
    """
    if accept in ("text/csv", "application/json"):
        # Convert to CSV
        out = io.StringIO()
        out.write("proba,pred\n")
        # Ensure native types
        for p, y in prediction:
            out.write(f"{float(p)},{int(y)}\n")
        body = out.getvalue()
        return body, "text/csv"
    raise ValueError(f"Unsupported accept: {accept}")
