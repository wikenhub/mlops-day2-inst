# sagemaker/code/train.py
# Skeleton for a SageMaker Training entrypoint.
# This script will:
#   1) Read CSVs from training/validation/test channels
#   2) Load preprocessing artifacts (fit in Lab 3)
#   3) Train a baseline Logistic Regression using class weights
#   4) Evaluate on val/test and save metrics
#   5) Save the trained model to /opt/ml/model/

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------
# 1) CLI arguments
# ---------------------------


def parse_args() -> argparse.Namespace:
    """
    Expose hyperparameters and common knobs so we can change behavior
    without editing code. Sensible defaults are provided.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Churn", help="Target label column name")
    p.add_argument("--max-iter", type=int, default=200, help="Max iterations for optimizer")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    p.add_argument("--penalty", default="l2", choices=["l2"], help="Regularization type")
    p.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga"],
        help="Optimizer (lbfgs is a good default for dense OHE features)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Reproducibility")
    p.add_argument(
        "--class-weights",
        default="auto",
        choices=["auto", "balanced", "none"],
        help=(
            "auto: use class_weights.json from artifacts if present, "
            "else fallback to 'balanced'. balanced: force sklearn 'balanced'. "
            "none: do not use class weights."
        ),
    )
    return p.parse_args()


# ---------------------------
# 2) IO paths (SageMaker Training)
# ---------------------------


def io_paths_from_env() -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Resolve Training job channels and output dirs. SageMaker sets these env vars.
    We also provide local defaults so you can dry-run outside SageMaker.
    """
    in_train = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    in_val = Path(os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"))
    in_test = Path(os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    in_art = Path(os.environ.get("SM_CHANNEL_ARTIFACTS", "/opt/ml/input/data/artifacts"))

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    for d in [model_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return in_train, in_val, in_test, in_art, model_dir, out_dir


# ---------------------------
# 3) Small helpers (CSV loading)
# ---------------------------


def read_all_csvs(folder: Path) -> pd.DataFrame:
    """
    Load one or more CSV files from a folder and concat them.
    Accepts both single-CSV and multi-sharded datasets.
    """
    files: list[Path] = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {folder}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate X (features) and y (labels). Validates that the target exists.
    """
    if target not in df.columns:
        raise KeyError(f"Expected target column '{target}' not found. Got: {list(df.columns)[:5]}…")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# ---------------------------
# 4) Load preprocessing artifacts (from Lab 3)
# ---------------------------


def load_artifacts(art_dir: Path):
    preproc_path = art_dir / "preprocess.joblib"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"Missing {preproc_path}. Run Lab 3 to generate preprocessing artifacts."
        )
    preproc = joblib.load(preproc_path)

    class_weights = None
    cw_path = art_dir / "class_weights.json"
    if cw_path.exists():
        try:
            class_weights = json.loads(cw_path.read_text())
        except Exception:
            class_weights = None

    return preproc, class_weights


# ---------------------------
# 5) Apply preprocessing and make numeric labels
# ---------------------------


def apply_preprocessing(preproc, X_raw: pd.DataFrame):
    """
    Transform raw features with the fitted pipeline.
    Some encoders (OneHotEncoder) return a sparse matrix; we densify for solvers
    that require dense arrays.
    """
    X = preproc.transform(X_raw)
    # If it's a scipy.sparse matrix, convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def map_labels_to_binary(y: pd.Series) -> np.ndarray:
    """
    Map 'Yes'/'No' (or 'yes'/'no') to 1/0 so sklearn metrics & models work.
    """
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
    y_mapped = y.map(mapping)
    if y_mapped.isna().any():
        raise ValueError("Target column contains unexpected values; expected Yes/No.")
    return y_mapped.astype(int).to_numpy()


# ---------------------------
# 6) Train a baseline Logistic Regression
# ---------------------------


def pick_class_weight(arg_choice: str, artifact_weights: dict | None):
    """
    arg_choice:
      - 'auto': use artifact weights if available, else sklearn 'balanced'
      - 'balanced': always use sklearn 'balanced'
      - 'none': do not use class weights
    """
    if arg_choice == "auto":
        return artifact_weights or "balanced"
    if arg_choice == "balanced":
        return "balanced"
    return None  # 'none'


def fit_logistic_regression(X_train, y_train, args, class_weights):
    cw = pick_class_weight(args.class_weights, class_weights)
    model = LogisticRegression(
        max_iter=args.max_iter,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,  # 'lbfgs' is a good default for dense inputs
        class_weight=cw,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# 7) Evaluation helpers
# ---------------------------


def evaluate_split(model, X, y, split_name: str) -> dict:
    """
    Compute common metrics. For a first pass we capture:
      - ROC AUC (area under ROC curve)
      - PR AUC (average precision = area under Precision-Recall)
      - Confusion matrix
      - Classification report (per-class precision/recall/F1)
    """

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "split": split_name,
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True),
    }


# ---------------------------
# 8) Main
# ---------------------------


def main():
    args = parse_args()
    in_train, in_val, in_test, in_art, model_dir, out_dir = io_paths_from_env()

    # Load CSVs from SageMaker channels
    df_train = read_all_csvs(in_train)
    df_val = read_all_csvs(in_val)
    df_test = read_all_csvs(in_test)

    # Split features/target
    Xtr_raw, ytr_raw = split_features_target(df_train, args.target)
    Xva_raw, yva_raw = split_features_target(df_val, args.target)
    Xte_raw, yte_raw = split_features_target(df_test, args.target)

    # Load fitted preprocessor + optional class weights
    preproc, cw = load_artifacts(in_art)

    # Transform features
    Xtr = apply_preprocessing(preproc, Xtr_raw)
    Xva = apply_preprocessing(preproc, Xva_raw)
    Xte = apply_preprocessing(preproc, Xte_raw)

    # Map labels to 0/1
    ytr = map_labels_to_binary(ytr_raw)
    yva = map_labels_to_binary(yva_raw)
    yte = map_labels_to_binary(yte_raw)

    # Fit model
    model = fit_logistic_regression(Xtr, ytr, args, cw)

    # Evaluate
    metrics = {
        "train": evaluate_split(model, Xtr, ytr, "train"),
        "val": evaluate_split(model, Xva, yva, "val"),
        "test": evaluate_split(model, Xte, yte, "test"),
        "params": {
            "solver": args.solver,
            "C": args.C,
            "penalty": args.penalty,
            "max_iter": args.max - iter if hasattr(args, "max-iter") else args.max_iter,  # guard
            "class_weights": args.class_weights,
        },
    }

    # Save metrics for easy retrieval
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save the trained model (and the preprocessor for convenience)
    #    Best practice for inference is to carry the preprocessor along,
    #    so a single artifact can transform + predict.
    joblib.dump({"model": model, "preprocess": preproc}, model_dir / "model.joblib")

    print("Saved metrics to", out_dir / "metrics.json")
    print("Saved model to", model_dir / "model.joblib")


if __name__ == "__main__":
    main()
