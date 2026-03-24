# -*- coding: utf-8 -*-
"""
feature_fixes.py
================
Applies 4 post-processing corrections to train_final.parquet and test_final.parquet
before the modeling phase.

Correcoes:
  1. bureau_max_overdue -> binary flag bureau_had_overdue + fill NaN with 0
  2. EXT_SOURCE_1 -> LightGBM model imputation (replacing naive median fill)
  3. Remove property columns with >60% nulls in raw data
  4. Validate DAYS_* for unexpected positive values (report only, no removal)

Usage:
    python src/features/feature_fixes.py
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_PATH = PROCESSED / "train_final.parquet"
TEST_PATH = PROCESSED / "test_final.parquet"
IMPUTER_PATH = MODELS_DIR / "imputer_ext_source_1.pkl"
REPORT_PATH = REPORTS_DIR / "feature_fix_report.md"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# --- Columns to drop (property with >60% nulls in raw data) ------------------
PROPERTY_COLS_TO_DROP = [
    "COMMONAREA_AVG", "COMMONAREA_MEDI", "COMMONAREA_MODE",
    "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAPARTMENTS_MODE",
    "FONDKAPREMONT_MODE",
    "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE",
    "YEARS_BUILD_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE",
]

# Features used by the EXT_SOURCE_1 imputation model
IMPUTER_FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
]

# Median value used in data-prep phase to fill EXT_SOURCE_1 nulls
MEDIAN_FILL_VALUE = 0.506503  # discovered from EDA of interim data


def load_data():
    """Load train and test parquet files."""
    print("Loading train and test datasets...")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape:  {test.shape}")
    return train, test


# --- Correction 1: bureau_max_overdue ----------------------------------------

def fix_bureau_max_overdue(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Create bureau_had_overdue binary flag.
    NaN in bureau_max_overdue treated as 0 (no overdue history = never overdue).
    """
    print("\n[Correction 1] bureau_max_overdue -> bureau_had_overdue flag")

    results = {}

    for name, df in [("train", train), ("test", test)]:
        # Fill any remaining NaN with 0 (safe default)
        df["bureau_max_overdue"] = df["bureau_max_overdue"].fillna(0)

        # Create binary flag
        df["bureau_had_overdue"] = (df["bureau_max_overdue"] > 0).astype(int)

        n_had = df["bureau_had_overdue"].sum()
        n_no = (df["bureau_had_overdue"] == 0).sum()
        pct = n_had / len(df) * 100
        print(f"  {name}: bureau_had_overdue=1 => {n_had:,} ({pct:.1f}%)"
              f"  |  =0 => {n_no:,} ({100-pct:.1f}%)")

        results[name] = {"had_overdue": int(n_had), "no_overdue": int(n_no),
                         "pct_had": round(pct, 2)}

    return train, test, results


# --- Correction 2: EXT_SOURCE_1 model imputation -----------------------------

def _encode_education(series: pd.Series) -> pd.Series:
    """Ordinal encode NAME_EDUCATION_TYPE for LightGBM."""
    order = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4,
    }
    return series.map(order).fillna(-1).astype(int)


def _prepare_imputer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and prepare features for the imputer model."""
    X = df[IMPUTER_FEATURES].copy()
    if "NAME_EDUCATION_TYPE" in X.columns:
        X["NAME_EDUCATION_TYPE"] = _encode_education(X["NAME_EDUCATION_TYPE"])
    # Remaining numeric NaNs: let LightGBM handle them natively
    return X


def _recover_null_mask_from_raw(split: str) -> set:
    """
    Recover which SK_ID_CURR had EXT_SOURCE_1 originally null in the raw CSV.
    This is the ground truth mask -- more reliable than detecting the median fill value.
    """
    raw_file = "application_train.csv" if split == "train" else "application_test.csv"
    raw_path = RAW / raw_file
    print(f"  Reading raw {raw_file} to recover original null mask...")
    raw = pd.read_csv(raw_path, usecols=["SK_ID_CURR", "EXT_SOURCE_1"])
    null_ids = raw.loc[raw["EXT_SOURCE_1"].isna(), "SK_ID_CURR"]
    return set(null_ids.values)


def fix_ext_source_1(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Replace the naive median imputation of EXT_SOURCE_1 with a LightGBM regressor
    trained on records that originally had EXT_SOURCE_1 present.
    """
    print("\n[Correction 2] EXT_SOURCE_1 -> LightGBM model imputation")

    results = {}

    # Step 1: recover original null masks from raw CSVs
    train_null_ids = _recover_null_mask_from_raw("train")
    test_null_ids = _recover_null_mask_from_raw("test")

    train_was_null = train["SK_ID_CURR"].isin(train_null_ids)
    test_was_null = test["SK_ID_CURR"].isin(test_null_ids)

    print(f"  Train: {train_was_null.sum():,} records with originally-null EXT_SOURCE_1"
          f" ({train_was_null.mean()*100:.1f}%)")
    print(f"  Test:  {test_was_null.sum():,} records with originally-null EXT_SOURCE_1"
          f" ({test_was_null.mean()*100:.1f}%)")

    # Step 2: prepare features
    X_train_all = _prepare_imputer_features(train)
    X_test_all = _prepare_imputer_features(test)

    # Records where EXT_SOURCE_1 was originally present (use as training data)
    mask_present = ~train_was_null
    X_fit = X_train_all[mask_present]
    y_fit = train.loc[mask_present, "EXT_SOURCE_1"]

    print(f"  Imputer training set size: {len(X_fit):,} records")

    # Step 3: train LightGBM regressor
    imputer = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    imputer.fit(X_fit, y_fit)

    # Cross-validation RMSE on training fold
    cv_scores = cross_val_score(
        imputer, X_fit, y_fit,
        scoring="neg_root_mean_squared_error",
        cv=5, n_jobs=-1
    )
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"  CV RMSE (5-fold): {cv_rmse:.4f} +/- {cv_std:.4f}")

    # Train RMSE
    y_pred_train_fit = imputer.predict(X_fit)
    train_rmse = float(np.sqrt(mean_squared_error(y_fit, y_pred_train_fit)))
    print(f"  Train RMSE: {train_rmse:.4f}")

    results["cv_rmse"] = round(float(cv_rmse), 4)
    results["cv_rmse_std"] = round(float(cv_std), 4)
    results["train_rmse"] = round(float(train_rmse), 4)
    results["n_fit"] = int(mask_present.sum())
    results["n_imputed_train"] = int(train_was_null.sum())
    results["n_imputed_test"] = int(test_was_null.sum())

    # Step 4: impute missing values in train
    if train_was_null.sum() > 0:
        X_to_impute_train = X_train_all[train_was_null]
        train.loc[train_was_null, "EXT_SOURCE_1"] = imputer.predict(X_to_impute_train)

    # Step 5: create imputation flag in train
    train["ext_source_1_imputed"] = train_was_null.astype(int).values

    # Step 6: apply same imputer to test (no re-training)
    if test_was_null.sum() > 0:
        X_to_impute_test = X_test_all[test_was_null]
        test.loc[test_was_null, "EXT_SOURCE_1"] = imputer.predict(X_to_impute_test)

    test["ext_source_1_imputed"] = test_was_null.astype(int).values

    # Step 7: save imputer model
    with open(IMPUTER_PATH, "wb") as f:
        pickle.dump(imputer, f)
    print(f"  Imputer saved: {IMPUTER_PATH}")

    print("  EXT_SOURCE_1 post-imputation stats (train):")
    print(f"    mean={train['EXT_SOURCE_1'].mean():.4f}  "
          f"std={train['EXT_SOURCE_1'].std():.4f}  "
          f"min={train['EXT_SOURCE_1'].min():.4f}  "
          f"max={train['EXT_SOURCE_1'].max():.4f}")

    return train, test, results


# --- Correction 3: Remove property columns with >60% nulls -------------------

def fix_drop_property_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Drop columns for property attributes that had >60% nulls in raw data."""
    print("\n[Correction 3] Drop property columns with >60% nulls")

    to_drop_train = [c for c in PROPERTY_COLS_TO_DROP if c in train.columns]
    to_drop_test = [c for c in PROPERTY_COLS_TO_DROP if c in test.columns]

    print(f"  Columns found in train: {len(to_drop_train)} -> dropping")
    print(f"  Columns found in test:  {len(to_drop_test)} -> dropping")
    print(f"  Columns: {to_drop_train}")

    train.drop(columns=to_drop_train, inplace=True, errors="ignore")
    test.drop(columns=to_drop_test, inplace=True, errors="ignore")

    results = {
        "cols_dropped_from_train": to_drop_train,
        "cols_dropped_from_test": to_drop_test,
        "n_dropped": len(to_drop_train),
    }

    print(f"  Train shape after drop: {train.shape}")
    print(f"  Test shape after drop:  {test.shape}")

    return train, test, results


# --- Correction 4: Data Leakage Validation (DAYS_*) --------------------------

def check_days_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Check all DAYS_* columns for unexpected positive values.
    Convention: DAYS_* should be <= 0 (days relative to application date).
    Positive values may indicate data leakage (future info) or encoding errors.
    Report only -- do not remove.
    """
    print("\n[Correction 4] Data leakage check -- DAYS_* positive values")

    days_cols = [c for c in train.columns if c.startswith("DAYS_")]
    results = {"columns_checked": days_cols, "findings": []}

    for col in days_cols:
        train_pos = int((train[col] > 0).sum())
        test_pos = int((test[col] > 0).sum()) if col in test.columns else 0

        train_min = float(train[col].min())
        train_max = float(train[col].max())

        status = "OK" if train_pos == 0 and test_pos == 0 else "ALERT"
        finding = {
            "column": col,
            "train_positive": train_pos,
            "test_positive": test_pos,
            "train_min": round(train_min, 2),
            "train_max": round(train_max, 2),
            "status": status,
        }
        results["findings"].append(finding)

        flag = " [ALERT]" if status == "ALERT" else ""
        print(f"  {col}: train_pos={train_pos}, test_pos={test_pos}, "
              f"min={train_min:.0f}, max={train_max:.0f}{flag}")

    total_alerts = sum(1 for f in results["findings"] if f["status"] == "ALERT")
    results["total_alerts"] = total_alerts
    print(f"  Summary: {total_alerts}/{len(days_cols)} DAYS_* columns have positive values")

    return results


# --- Report generation -------------------------------------------------------

def generate_report(
    shape_before_train, shape_before_test,
    shape_after_train, shape_after_test,
    c1_results, c2_results, c3_results, c4_results,
    final_columns,
):
    """Write feature_fix_report.md."""

    lines = []
    lines.append("# Feature Fix Report")
    lines.append("")
    lines.append("**Data:** 2026-03-23")
    lines.append("**Status:** Concluido")
    lines.append("")

    # Dataset shape before/after
    lines.append("## Shape Before/After Corrections")
    lines.append("")
    lines.append("| Dataset | Before | After | Delta cols |")
    lines.append("|---------|--------|-------|------------|")
    delta_train = shape_after_train[1] - shape_before_train[1]
    delta_test = shape_after_test[1] - shape_before_test[1]
    lines.append(f"| train_final.parquet | {shape_before_train} | {shape_after_train} | {delta_train:+d} |")
    lines.append(f"| test_final.parquet  | {shape_before_test}  | {shape_after_test}  | {delta_test:+d}  |")
    lines.append("")

    # Correction 1
    lines.append("## Correction 1 - bureau_had_overdue Flag")
    lines.append("")
    lines.append("Created `bureau_had_overdue` binary feature from `bureau_max_overdue`.")
    lines.append("Logic: `1` if `bureau_max_overdue > 0`, else `0`. NaN treated as `0` (no overdue history).")
    lines.append("")
    lines.append("### Distribution (Train)")
    lines.append("")
    lines.append("| Value | Count | % |")
    lines.append("|-------|-------|---|")
    r = c1_results["train"]
    lines.append(f"| 1 (had overdue) | {r['had_overdue']:,} | {r['pct_had']:.1f}% |")
    lines.append(f"| 0 (no overdue)  | {r['no_overdue']:,} | {100-r['pct_had']:.1f}% |")
    lines.append("")
    lines.append("### Distribution (Test)")
    lines.append("")
    lines.append("| Value | Count | % |")
    lines.append("|-------|-------|---|")
    r = c1_results["test"]
    lines.append(f"| 1 (had overdue) | {r['had_overdue']:,} | {r['pct_had']:.1f}% |")
    lines.append(f"| 0 (no overdue)  | {r['no_overdue']:,} | {100-r['pct_had']:.1f}% |")
    lines.append("")

    # Correction 2
    lines.append("## Correction 2 - EXT_SOURCE_1 Model Imputation")
    lines.append("")
    lines.append("Replaced naive median imputation with a **LightGBM regressor**.")
    lines.append("")
    lines.append(f"- Original nulls in train: **{c2_results['n_imputed_train']:,}** records "
                 f"({c2_results['n_imputed_train']/shape_before_train[0]*100:.1f}%)")
    lines.append(f"- Original nulls in test: **{c2_results['n_imputed_test']:,}** records "
                 f"({c2_results['n_imputed_test']/shape_before_test[0]*100:.1f}%)")
    lines.append(f"- Imputer trained on: **{c2_results['n_fit']:,}** records with EXT_SOURCE_1 present")
    lines.append("")
    lines.append("### Imputer Features")
    lines.append("")
    lines.append("| Feature |")
    lines.append("|---------|")
    for feat in IMPUTER_FEATURES:
        lines.append(f"| `{feat}` |")
    lines.append("")
    lines.append("### Model Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| CV RMSE (5-fold) | {c2_results['cv_rmse']:.4f} +/- {c2_results['cv_rmse_std']:.4f} |")
    lines.append(f"| Train RMSE       | {c2_results['train_rmse']:.4f} |")
    lines.append("")
    lines.append("Imputer model saved: `models/imputer_ext_source_1.pkl`")
    lines.append("Flag `ext_source_1_imputed` added to train and test.")
    lines.append("")

    # Correction 3
    lines.append("## Correction 3 - Property Columns Removed (>60% nulls)")
    lines.append("")
    lines.append(f"Removed **{c3_results['n_dropped']}** columns from train and test.")
    lines.append("")
    lines.append("| Column |")
    lines.append("|--------|")
    for col in c3_results["cols_dropped_from_train"]:
        lines.append(f"| `{col}` |")
    lines.append("")

    # Correction 4
    lines.append("## Correction 4 - Data Leakage Check (DAYS_* columns)")
    lines.append("")
    lines.append("Convention: DAYS_* values should be <= 0 (days before application date).")
    lines.append("Positive values would indicate future information (potential leakage).")
    lines.append("")
    lines.append("| Column | Train Positive | Test Positive | Train Min | Train Max | Status |")
    lines.append("|--------|---------------|---------------|-----------|-----------|--------|")
    for f in c4_results["findings"]:
        status_icon = "OK" if f["status"] == "OK" else "ALERT"
        lines.append(f"| `{f['column']}` | {f['train_positive']:,} | {f['test_positive']:,} "
                     f"| {f['train_min']:.0f} | {f['train_max']:.0f} | {status_icon} |")
    lines.append("")
    lines.append(f"**Result:** {c4_results['total_alerts']}/{len(c4_results['columns_checked'])} "
                 f"DAYS_* columns with positive values detected.")
    if c4_results["total_alerts"] == 0:
        lines.append("No data leakage detected in DAYS_* features.")
    else:
        lines.append("**Action required:** Review columns flagged as ALERT before modeling.")
    lines.append("")

    # Final feature list
    lines.append("## Final Feature List")
    lines.append("")
    lines.append(f"Total features in train_final.parquet: **{len(final_columns)}**")
    lines.append("")
    lines.append("```")
    for col in sorted(final_columns):
        lines.append(col)
    lines.append("```")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved: {REPORT_PATH}")


# --- Main --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Feature Fix Pipeline")
    print("=" * 60)

    # Load
    train, test = load_data()
    shape_before_train = train.shape
    shape_before_test = test.shape

    # Correction 1
    train, test, c1_results = fix_bureau_max_overdue(train, test)

    # Correction 2
    train, test, c2_results = fix_ext_source_1(train, test)

    # Correction 3
    train, test, c3_results = fix_drop_property_columns(train, test)

    # Correction 4 (report only)
    c4_results = check_days_leakage(train, test)

    # Save corrected datasets
    print("\nSaving corrected datasets...")
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    print(f"  train_final.parquet: {TRAIN_PATH}  shape={train.shape}")
    print(f"  test_final.parquet:  {TEST_PATH}   shape={test.shape}")

    # Generate report
    generate_report(
        shape_before_train, shape_before_test,
        train.shape, test.shape,
        c1_results, c2_results, c3_results, c4_results,
        list(train.columns),
    )

    print("\n" + "=" * 60)
    print("All corrections applied successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
