# sagemaker/code/train.py
# Skeleton for a SageMaker Training entrypoint.
# This script will:
#   1) Read CSVs from training/validation/test channels
#   2) Load preprocessing artifacts (fit in Lab 4)
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

# --- args: keep near the top of train.py ---
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
    p.add_argument("-C", "--C", type=float, default=1.0, help="Inverse regularization strength")
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
            "auto: use class_weights from preprocess.joblib if present, "
            "else fallback to sklearn 'balanced'. "
            "balanced: force sklearn 'balanced'. none: do not use class weights."
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
# 3) Schema & CSV loading helpers (for processed splits)
# ---------------------------
# Assumes: import json, pandas as pd, and from pathlib import Path are already at top.


def read_schema_columns(art_dir: Path) -> tuple[list[str], str, dict[str, int]]:
    """
    Read columns.json/schema.json if present to discover feature order,
    label name, and mapping.
    """
    feature_cols: list[str] = []
    label_name: str = "label"
    label_mapping: dict[str, int] = {"No": 0, "Yes": 1}

    cols_file = art_dir / "columns.json"
    sch_file = art_dir / "schema.json"

    if cols_file.exists():
        obj = json.loads(cols_file.read_text())
        cols = obj.get("columns", [])
        if isinstance(cols, list) and cols:
            # Our convention is f0..fN + "label"
            if cols[-1] == "label":
                feature_cols = [c for c in cols if c != "label"]
                label_name = "label"

    if sch_file.exists():
        sch = json.loads(sch_file.read_text())
        label_name = sch.get("label_column", label_name)
        label_mapping = sch.get("label_mapping", label_mapping)

    return feature_cols, label_name, label_mapping


def read_all_csvs(folder: Path) -> pd.DataFrame:
    """Load one or more CSV files from a folder and concat them."""
    files: list[Path] = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {folder}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def load_processed_split(
    folder: Path, feature_cols: list[str], label_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a processed split and return (X, y).
    If feature_cols is empty, infer as all columns except label_name
    (or fall back to "all but last" if label_name missing).
    """
    df = read_all_csvs(folder)

    if not feature_cols:
        if label_name in df.columns:
            feature_cols = [c for c in df.columns if c != label_name]
        else:
            # Fallback: assume last column is label
            feature_cols = df.columns[:-1].tolist()
            label_name = df.columns[-1]

    if label_name not in df.columns:
        raise KeyError(
            f"Label column '{label_name}' not found in {folder}. "
            f"Available: {list(df.columns)[:8]}..."
        )

    X = df[feature_cols].astype("float64")
    y = df[label_name].astype("int64")
    return X, y


# ---------------------------
# 4) Load bundle (preprocess + metadata + class weights)
# ---------------------------


def load_artifacts_bundle(art_dir: Path) -> dict:
    """
    Load the single bundle produced by preprocessing.
    Expected keys include:
      - 'preprocessor' (fitted ColumnTransformer)
      - 'scaler'
      - 'class_weights' (dict like {0: w0, 1: w1})
      - 'feature_names' / 'feature_count' (best effort)
    """
    path = art_dir / "preprocess.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Lab 4 to generate preprocessing artifacts.")
    return joblib.load(path)


# ---------------------------
# 5) Light type guards (processed data)
# ---------------------------


def ensure_numeric(X: pd.DataFrame) -> np.ndarray:
    arr = X.astype("float64").to_numpy()
    return arr


def ensure_binary_int(y: pd.Series) -> np.ndarray:
    # Processed splits already have 0/1 in 'label', but be strict about dtype.
    yy = pd.to_numeric(y, errors="raise").astype("int64")
    if set(pd.unique(yy)) - {0, 1}:
        raise ValueError("Label must be binary 0/1 in processed splits.")
    return yy.to_numpy()


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
    from sklearn.metrics import (
        roc_auc_score,
    )

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


def resolve_class_weight(
    mode: str, y_train: pd.Series, bundle: dict | None
) -> dict[int, float] | None:
    """
    - 'none'     -> None
    - 'balanced' -> sklearn-computed from y_train
    - 'auto'     -> bundle['class_weights'] if present else sklearn 'balanced'
    """
    from sklearn.utils.class_weight import compute_class_weight

    if mode == "none":
        return None
    if mode == "balanced":
        classes = np.sort(y_train.unique())
        weights = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
        return {int(c): float(w) for c, w in zip(classes, weights)}
    # auto
    if bundle and "class_weights" in bundle:
        cw = bundle["class_weights"]
        return {int(k): float(v) for k, v in cw.items()}
    classes = np.sort(y_train.unique())
    weights = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
    return {int(c): float(w) for c, w in zip(classes, weights)}


def main():
    args = parse_args()
    in_train, in_val, in_test, in_art, model_dir, out_dir = io_paths_from_env()

    # 1) Read schema & columns (so we know features and label names)
    feature_cols, label_name, _ = read_schema_columns(in_art)

    # 2) Load processed splits
    Xtr_df, ytr_s = load_processed_split(in_train, feature_cols, label_name)
    Xva_df, yva_s = load_processed_split(in_val, feature_cols, label_name)
    Xte_df, yte_s = load_processed_split(in_test, feature_cols, label_name)

    # 3) Enforce numeric types (already numeric, but be explicit)
    Xtr, Xva, Xte = ensure_numeric(Xtr_df), ensure_numeric(Xva_df), ensure_numeric(Xte_df)
    ytr, yva, yte = ensure_binary_int(ytr_s), ensure_binary_int(yva_s), ensure_binary_int(yte_s)

    # 4) Load artifacts bundle (for class weights & for future inference packaging)
    bundle = load_artifacts_bundle(in_art)
    class_weight = resolve_class_weight(args.class_weights, ytr_s, bundle)

    # 5) Train
    model = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state,
        class_weight=class_weight,
    )
    model.fit(Xtr, ytr)

    # 6) Evaluate
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    proba_va = model.predict_proba(Xva)[:, 1]
    pred_va = (proba_va >= 0.5).astype("int64")
    prec, rec, f1, _ = precision_recall_fscore_support(
        yva, pred_va, average="binary", zero_division=0
    )

    metrics = {
        "val/roc_auc": float(roc_auc_score(yva, proba_va)),
        "val/pr_auc": float(average_precision_score(yva, proba_va)),
        "val/accuracy": float(accuracy_score(yva, pred_va)),
        "val/f1": float(f1),
        "val/precision": float(prec),
        "val/recall": float(rec),
    }

    proba_te = model.predict_proba(Xte)[:, 1]
    pred_te = (proba_te >= 0.5).astype("int64")
    metrics.update(
        {
            "test/roc_auc": float(roc_auc_score(yte, proba_te)),
            "test/pr_auc": float(average_precision_score(yte, proba_te)),
            "test/accuracy": float(accuracy_score(yte, pred_te)),
            "test/f1": float(f1_score(yte, pred_te)),
        }
    )

    # 7) Save metrics and model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # also drop a copy next to the model artifact
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # (Nice for later inference labs): ship the preprocessor in the same artifact
    to_save = {"model": model}
    if "preprocessor" in bundle:
        to_save["preprocess"] = bundle["preprocessor"]
    joblib.dump(to_save, model_dir / "model.joblib")

    print("=== Training complete ===")
    print(json.dumps(metrics, indent=2))
    print("Saved:", model_dir / "model.joblib", "and", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
