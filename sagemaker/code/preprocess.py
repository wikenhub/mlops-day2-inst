# Import statements at top

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Churn")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def io_paths_from_env():
    # These env vars are set automatically by SageMaker Processing.
    # Defaults allow local dry-runs.
    input_dir = Path(os.environ.get("SM_INPUT_DIR", "/opt/ml/processing/input"))
    out_train = Path(os.environ.get("SM_OUTPUT_TRAIN_DIR", "/opt/ml/processing/train"))
    out_val = Path(os.environ.get("SM_OUTPUT_VAL_DIR", "/opt/ml/processing/val"))
    out_test = Path(os.environ.get("SM_OUTPUT_TEST_DIR", "/opt/ml/processing/test"))
    out_art = Path(os.environ.get("SM_OUTPUT_ARTIFACT_DIR", "/opt/ml/processing/artifacts"))
    out_rep = Path(os.environ.get("SM_OUTPUT_REPORT_DIR", "/opt/ml/processing/report"))
    for d in [out_train, out_val, out_test, out_art, out_rep]:
        d.mkdir(parents=True, exist_ok=True)
    return input_dir, out_train, out_val, out_test, out_art, out_rep


def _csv_paths(root: Path) -> list[Path]:
    """Return all CSV-like files under root, sorted for deterministic order."""
    files = sorted([*root.rglob("*.csv"), *root.rglob("*.csv.gz")])
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files


def load_telco_dataframe(input_dir: Path) -> pd.DataFrame:
    """Load one or many CSVs, normalize schema, and apply Telco-specific fixes."""
    paths = _csv_paths(input_dir)

    # 1) Load all parts
    frames = []
    for p in paths:
        df_part = pd.read_csv(p)
        frames.append(df_part)

    # 2) Union columns across parts, then concat (keeps missing columns as NaN)
    all_cols = sorted(set().union(*(f.columns for f in frames)))
    frames = [f.reindex(columns=all_cols) for f in frames]
    df = pd.concat(frames, ignore_index=True, sort=False)

    # 3) Telco-specific cleanups
    #    a) 'customerID' is an identifier, not a predictive feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    #    b) 'TotalCharges' sometimes arrives as string; coerce to numeric (bad parses -> NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    #    c) Trim whitespace in all object columns to avoid "Yes " vs "Yes"
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    # 4) Basic hygiene: drop fully empty rows (rare, but harmless)
    df = df.dropna(how="all")

    # 5) Sanity check: ensure target exists
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in input data.")

    return df


def eda_summary(df: pd.DataFrame, target: str) -> dict:
    """
    Build a compact, serializable EDA report.

    Why this shape:
    - Small enough to store per run (e.g., JSON in S3).
    - Focused on fields you can quickly compare across runs for drift/debug.
    """
    # 1) Identify numeric vs categorical columns (keep the target out of cat list)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target]

    # 2) Basic data quality signals
    null_counts = df.isna().sum().to_dict()  # missing values per column
    nunique = df.nunique(dropna=True).to_dict()  # cardinality per column

    # 3) Outlier counts for numeric columns using Tukey's IQR rule
    #    (robust to non-normal data; fast to compute)
    outliers_iqr_count = {}
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            outliers_iqr_count[c] = 0
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            outliers_iqr_count[c] = 0
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr_count[c] = int(((s < lower) | (s > upper)).sum())

    # 4) Class balance for the target (keep raw values; mapping to 0/1 happens later)
    target_counts = df[target].value_counts(dropna=False).to_dict()

    # 5) Package into a JSON-friendly dict
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "null_counts": null_counts,
        "nunique": nunique,
        "outliers_iqr_count": outliers_iqr_count,
        "target": target,
        "target_counts": target_counts,
    }


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
):
    """
    Split into train/val/test with stratification on y.

    Why two-step split?
    - First: train vs (val+test) using (test_size + val_size).
    - Second: split that holdout into val and test using the *relative* ratio.

    Notes:
    - Stratification keeps the positive/negative churn ratio similar across splits.
    - If stratification is impossible (e.g., too few samples in a class), we fall back
      to non-stratified splits with a clear warning.
    """
    if test_size < 0 or val_size < 0 or (test_size + val_size) >= 1.0:
        raise ValueError("Require 0 <= test_size, val_size and (test_size + val_size) < 1.0")

    holdout = test_size + val_size
    rel_val = val_size / holdout if holdout > 0 else 0.0

    # Helper to attempt a stratified split, then fallback if it fails
    def _safe_split(Xa, ya, **kwargs):
        try:
            return train_test_split(Xa, ya, stratify=ya, **kwargs)
        except ValueError as e:
            # Common when a class has very few rows; continue without stratification.
            print(f"[WARN] Stratified split failed: {e}. Falling back to non-stratified split.")
            return train_test_split(Xa, ya, **{k: v for k, v in kwargs.items() if k != "stratify"})

    # 1) Train vs (Val+Test)
    X_train, X_temp, y_train, y_temp = _safe_split(
        X, y, test_size=holdout, random_state=random_state
    )

    # 2) (Val+Test) -> Val vs Test
    X_val, X_test, y_val, y_test = _safe_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=random_state
    )

    # (Optional) Align indices to simple 0..n-1 for neat downstream concatenations
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame):
    """
    Create a reusable preprocessing pipeline:
      - Numeric: median impute  → (later) standard scale
      - Categorical: most-frequent impute → one-hot encode (handle_unknown='ignore')

    Returns:
      preprocessor: ColumnTransformer that imputes both blocks and encodes categoricals
      scaler: StandardScaler that we will apply only to the numeric block AFTER transform
      num_cols: list of numeric column names (to locate the numeric block)
      cat_cols: list of categorical column names
    """
    # 1) Partition feature columns by dtype
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 2) Define per-block pipelines
    #    Numeric block: just impute here; we will scale AFTER the whole transform
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    #    Categorical block: impute then one-hot encode.
    #    - handle_unknown='ignore' keeps the pipeline stable if new categories
    #    - appear at inference time
    #    - sparse_output=False yields a dense matrix (easier to save as CSV later)
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3) Column-wise assembly
    #    Put numeric FIRST so we know its block is at the start of the transformed matrix.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense output from the ColumnTransformer
        verbose_feature_names_out=False,  # cleaner output names if we ever inspect them
    )

    # 4) Scaler to apply only to the numeric slice after transform
    #    with_mean=False is safe when the overall matrix might be treated as sparse-like;
    #    it also avoids centering which can be unnecessary for one-hot parts.
    scaler = StandardScaler(with_mean=False)

    return preprocessor, scaler, num_cols, cat_cols


def fit_and_transform(
    preprocessor,
    scaler: StandardScaler,
    num_cols: list[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit transforms on TRAIN only; apply to VAL/TEST.
    We arranged the ColumnTransformer as [numeric, categorical], so the
    first len(num_cols) columns of the transformed arrays are numeric.
    We scale ONLY that numeric slice (no scaling for one-hot features).
    """
    # 1) Fit imputers/encoders on TRAIN, then transform all splits
    Xt_train = preprocessor.fit_transform(X_train)  # shape: [n_train, n_features_after_encoding]
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)

    # 2) Scale only the numeric slice at the front (if any numerics exist)
    n_num = len(num_cols)
    if n_num > 0:
        # Fit scaler on TRAIN numerics, then apply to VAL/TEST numerics
        Xt_train[:, :n_num] = scaler.fit_transform(Xt_train[:, :n_num])
        Xt_val[:, :n_num] = scaler.transform(Xt_val[:, :n_num])
        Xt_test[:, :n_num] = scaler.transform(Xt_test[:, :n_num])

    return Xt_train, Xt_val, Xt_test


def compute_class_weights(y_train: pd.Series) -> dict[int, float]:
    """
    Compute per-class weights for imbalanced data.
    Returns a small dict like {0: w0, 1: w1} that we can pass to training.
    """
    # Ensure labels are integers (0/1). If strings, map them here.
    if y_train.dtype == "object":
        y_train = y_train.map({"No": 0, "Yes": 1}).astype("int64")

    classes_present = np.sort(y_train.unique())
    # If a class is missing in the current training fold, default to weight 1.0 for present classes.
    if len(classes_present) < 2:
        return {int(classes_present[0]): 1.0}

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_train.values,
    )
    return {int(c): float(w) for c, w in zip(classes_present, weights)}


def save_split(Xm, ym, path: Path):
    """
    Persist a split to CSV with columns: f0..fN, label.
    - Xm: 2D numpy-like array of features (after encoding/scaling)
    - ym: 1D pandas Series or array of labels (0/1)
    - path: destination CSV path
    """
    # Ensure arrays, align shapes, and build column names
    Xm = np.asarray(Xm)
    yv = np.asarray(ym).reshape(-1, 1)

    if Xm.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {Xm.shape}")
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError(f"Row mismatch: X has {Xm.shape[0]} rows, y has {yv.shape[0]}")

    cols = [f"f{i}" for i in range(Xm.shape[1])] + ["label"]
    data = np.hstack([Xm, yv])

    df_out = pd.DataFrame(data, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)


def persist_artifacts(
    out_art: Path,
    preprocessor,
    scaler,
    num_cols: list[str],
    cat_cols: list[str],
    class_weights: dict[int, float],
    feature_count: int,
):
    """
    Save fitted preprocessing objects + metadata as a single joblib bundle.
    Also try to record feature names for debugging/inspection.
    """
    # Try to recover feature names from the fitted ColumnTransformer.
    # Falls back to generic f0..fN if unavailable.
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(feature_count)]

    bundle = {
        "preprocessor": preprocessor,  # fitted ColumnTransformer (imputers + one-hot)
        "scaler": scaler,  # fitted StandardScaler (numeric slice)
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "feature_names": feature_names,
        "feature_count": int(feature_count),
        "class_weights": class_weights,
        "version": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    out_art.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_art / "preprocess.joblib")


def write_report(report_dict: dict, out_rep: Path, filename: str = "report.json"):
    """
    Serialize a small, human-readable JSON report (EDA + split sizes, etc.).
    """
    out_rep.mkdir(parents=True, exist_ok=True)
    (out_rep / filename).write_text(json.dumps(report_dict, indent=2))


def write_schema_files(
    out_art: Path,
    feature_names_after_encoding: list[str],
    raw_target_name: str,
    label_name: str,
    label_mapping: dict,
) -> None:
    """
    Emit simple, human+machine friendly schema files next to preprocess.joblib.
    - columns.json: exact column order in the emitted CSVs (f0..fN + label)
    - schema.json: minimal metadata the trainer (and humans) can rely on
    """
    out_art.mkdir(parents=True, exist_ok=True)

    # 1) Columns file matches the CSVs we emit: f0..fN + label
    fcols = [f"f{i}" for i in range(len(feature_names_after_encoding))]
    (out_art / "columns.json").write_text(json.dumps({"columns": fcols + [label_name]}, indent=2))

    # 2) Minimal but useful schema for audits and downstream consumers
    schema = {
        "raw_target": raw_target_name,  # e.g., "Churn" in raw data
        "label_column": label_name,  # "label" in our CSVs
        "feature_count": len(feature_names_after_encoding),
        "feature_names_after_encoding": feature_names_after_encoding,  # best effort
        "label_mapping": label_mapping,  # e.g., {"No": 0, "Yes": 1}
        "contract": "features are f0..fN (float), label is int {0,1}",
    }
    (out_art / "schema.json").write_text(json.dumps(schema, indent=2))


def extract_features_and_label(df: pd.DataFrame, target: str):
    """
    Split the DataFrame into features X and binary label y (0/1).

    - Accepts common string labels ('Yes'/'No', 'True'/'False', '1'/'0') case-insensitively.
    - Drops rows where the target is missing or un-mappable.
    - Returns (X, y) where y is int64 {0,1}.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Map common textual labels to integers
    if df[target].dtype == "object":
        mapping = {"no": 0, "yes": 1, "false": 0, "true": 1, "0": 0, "1": 1}
        y = df[target].astype(str).str.lower().map(mapping)
    elif df[target].dtype == bool:
        y = df[target].astype("int64")
    else:
        # numeric-ish: coerce and check it is binary
        y = pd.to_numeric(df[target], errors="coerce")
        unique = set(pd.unique(y.dropna()))
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Target '{target}' must be binary (0/1 or Yes/No). "
                f"Found values: {sorted(list(unique))[:10]}"
            )

    # Drop rows where target is NaN after mapping/coercion
    mask = y.notna()
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[WARN] Dropping {dropped} rows with invalid/missing target labels")
    y = y.loc[mask].astype("int64")
    X = df.loc[mask].drop(columns=[target])

    return X, y


def main():
    # 0) Parse args and resolve I/O paths
    args = parse_args()
    input_dir, out_train, out_val, out_test, out_art, out_rep = io_paths_from_env()

    # 1) Load raw data (supports many CSV parts) and apply Telco fixes
    df_raw = load_telco_dataframe(input_dir)

    # 2) Lightweight EDA summary (human review)
    report = eda_summary(df_raw, args.target)

    # 3) Extract features (X) and label (y as 0/1)
    X_all, y_all = extract_features_and_label(df_raw, args.target)

    # 4) Build preprocessing stack (impute/encode + later scaling for numerics)
    preprocessor, scaler, num_cols, cat_cols = build_preprocessor(X_all)

    # 5) Train/Val/Test split (stratified), then fit on TRAIN only
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(
        X_all,
        y_all,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    Xt_tr, Xt_va, Xt_te = fit_and_transform(preprocessor, scaler, num_cols, X_tr, X_va, X_te)

    # 6) Handle class imbalance (weights from TRAIN)
    class_w = compute_class_weights(y_tr)
    feature_count = int(Xt_tr.shape[1])

    # 7) Recover human-readable feature names if the transformer provides them
    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(feature_count)]

    # 8) Save splits (CSV) and artifacts (joblib bundle)
    save_split(Xt_tr, y_tr, out_train / "train.csv")
    save_split(Xt_va, y_va, out_val / "val.csv")
    save_split(Xt_te, y_te, out_test / "test.csv")
    persist_artifacts(
        out_art=out_art,
        preprocessor=preprocessor,
        scaler=scaler,
        num_cols=num_cols,
        cat_cols=cat_cols,
        class_weights=class_w,
        feature_count=feature_count,
    )

    # 9) write explicit schema files the trainer can rely on
    # We used extract_features_and_label() to map "Yes/No" → 1/0.
    # Recreate that mapping for the report/schema (students can see it).
    label_mapping = {"No": 0, "Yes": 1}
    write_schema_files(
        out_art=out_art,
        feature_names_after_encoding=feat_names,
        raw_target_name=args.target,  # e.g., "Churn" in raw data
        label_name="label",  # standardized output label name
        label_mapping=label_mapping,
    )

    # 10) Finalize and write report.json
    report.update(
        {
            "splits": {
                "train_rows": int(len(y_tr)),
                "val_rows": int(len(y_va)),
                "test_rows": int(len(y_te)),
            },
            "class_weights": class_w,
            "feature_count_after_encoding": feature_count,
            "params": {
                "target": args.target,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "random_state": args.random_state,
            },
        }
    )
    write_report(report, out_rep)

    # 11) Friendly console summary
    print("=== Processing complete ===")
    print(f"Train: {out_train / 'train.csv'}")
    print(f"Val:   {out_val / 'val.csv'}")
    print(f"Test:  {out_test / 'test.csv'}")
    print(f"Artifacts: {out_art / 'preprocess.joblib'}")
    print(f"Report:    {out_rep / 'report.json'}")


if __name__ == "__main__":
    main()
