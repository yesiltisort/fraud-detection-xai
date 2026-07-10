# -*- coding: utf-8 -*-


import os, re, sys, warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.sparse import issparse

import shap
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ===== baseline-compatible split settings =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000
ANON_PATTERNS = [r"^V\d+$", r"^C\d+$", r"^D\d+$", r"^M\d+$", r"^id_\d+$"]

def log(msg: str): print(f"[info] {msg}")

# ---------- data helpers ----------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df

def drop_anonymized_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if any(re.match(p, c) for p in ANON_PATTERNS)]
    return df.drop(columns=to_drop, errors="ignore")

def rebuild_splits(base: Path):
    tx = pd.read_csv(base / "train_transaction.csv")
    idf = pd.read_csv(base / "train_identity.csv")
    df = standardize_cols(tx.merge(idf, on="TransactionID", how="left"))
    df = drop_anonymized_cols(df)

    if "isFraud" not in df.columns:
        raise RuntimeError("'isFraud' column not found.")

    y_full = df["isFraud"].astype(int)
    X_full = df.drop(columns=["isFraud"], errors="ignore")

    if SUBSET_SIZE < len(df):
        keep_frac = SUBSET_SIZE / len(df)
        sss_keep = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - keep_frac), random_state=RANDOM_STATE)
        keep_idx, _ = next(sss_keep.split(X_full, y_full))
        df = df.iloc[keep_idx].reset_index(drop=True)
        log(f"Downsampled: {len(y_full)} -> {len(df)}")

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="ignore", downcast="float")
    X = X.apply(pd.to_numeric, errors="ignore", downcast="integer")

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION_OF_TEMP, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    return X_train, y_train.reset_index(drop=True), X_val, y_val.reset_index(drop=True)

def to_dense(preprocessor, X):
    Xt = preprocessor.transform(X)
    return Xt.toarray() if issparse(Xt) else Xt

# ---------- feature name extraction ----------
def get_feature_names_from_ct(preprocessor, input_df: pd.DataFrame) -> List[str]:
    """Try to get expanded feature names from ColumnTransformer; robust fallback."""
    # Fast path (sklearn >= 1.0)
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        pass

    feature_names: List[str] = []
    used_cols = []

    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop":
            continue


        final_step = trans
        try:
            if isinstance(trans, Pipeline):
                final_step = trans.steps[-1][1]
        except Exception:
            pass

        if cols is None:
            cols = [c for c in input_df.columns if c not in used_cols]
        else:
            if isinstance(cols, (list, tuple)):
                used_cols.extend(cols)

        if isinstance(final_step, OneHotEncoder):
            if isinstance(cols, (list, tuple)):
                input_features = list(cols)
                cats_list = final_step.categories_
                for i, col in enumerate(input_features):
                    cats = cats_list[i]
                    feature_names.extend([f"{col}__{str(cat)}" for cat in cats])
            else:
                
                input_features = input_df.columns[cols].tolist()
                cats_list = final_step.categories_
                for i, col in enumerate(input_features):
                    cats = cats_list[i]
                    feature_names.extend([f"{col}__{str(cat)}" for cat in cats])
        else:
            if isinstance(cols, (list, tuple)):
                feature_names.extend(list(cols))
            else:
                feature_names.extend(input_df.columns[cols].tolist())


    dim = to_dense(preprocessor, input_df.iloc[:1]).shape[1]
    if len(feature_names) != dim:
        feature_names = [f"f{i}" for i in range(dim)]
    return feature_names

def build_original_groups(preprocessor, input_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, List[int]]:
    
    groups: Dict[str, List[int]] = {}
    idx = 0

    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop":
            continue

        final_step = trans
        try:
            if isinstance(trans, Pipeline):
                final_step = trans.steps[-1][1]
        except Exception:
            pass

        if cols is None:
            cols = [c for c in input_df.columns if c not in [cc for _, _, cc in preprocessor.transformers_ if isinstance(cc, (list, tuple)) for cc in cc]]

        if isinstance(final_step, OneHotEncoder):
            if isinstance(cols, (list, tuple)):
                cats_list = final_step.categories_
                for i, col in enumerate(cols):
                    width = len(cats_list[i])
                    groups.setdefault(col, []).extend(range(idx, idx + width))
                    idx += width
            else:
                # indices case
                feat_names = input_df.columns[cols].tolist()
                cats_list = final_step.categories_
                for i, col in enumerate(feat_names):
                    width = len(cats_list[i])
                    groups.setdefault(col, []).extend(range(idx, idx + width))
                    idx += width
        else:
            if isinstance(cols, (list, tuple)):
                for col in cols:
                    groups.setdefault(col, []).append(idx); idx += 1
            else:
                for col in input_df.columns[cols].tolist():
                    groups.setdefault(col, []).append(idx); idx += 1

    dim = to_dense(preprocessor, input_df.iloc[:1]).shape[1]
    if idx < dim:
        for j in range(idx, dim):
            groups.setdefault(f"extra_{j}", []).append(j)
    return groups

# ---------- plotting helpers ----------
def plot_beeswarm_and_bar(shap_vals: np.ndarray, X_df: pd.DataFrame, beeswarm_path: Path, bar_path: Path):
    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_vals, X_df, feature_names=X_df.columns.tolist(), show=False, max_display=15)
    plt.tight_layout(); plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight"); plt.close()
    log(f"Saved: {beeswarm_path.resolve()}")

    # Bar (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_vals, X_df, feature_names=X_df.columns.tolist(), show=False, plot_type="bar", max_display=15)
    plt.tight_layout(); plt.savefig(bar_path, dpi=200, bbox_inches="tight"); plt.close()
    log(f"Saved: {bar_path.resolve()}")

def plot_grouped_bar(shap_vals: np.ndarray, groups: Dict[str, List[int]], out_path: Path, topN: int = 15):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    rows = [{"original_feature": k, "sum_mean_abs_shap": float(mean_abs[v].sum())} for k, v in groups.items()]
    df = pd.DataFrame(rows).sort_values("sum_mean_abs_shap", ascending=False)
    sub = df.head(topN)[::-1]

    plt.figure(figsize=(6, 10))
    plt.barh(sub["original_feature"], sub["sum_mean_abs_shap"])
    plt.xlabel("sum of mean |SHAP| (grouped to original feature)")
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    log(f"Saved: {out_path.resolve()}")

# ---------- SHAP wrappers ----------
def tree_shap(model, X_train_dense: np.ndarray, X_val_dense: np.ndarray):
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(
        X_train_dense.shape[0],
        size=min(1000, X_train_dense.shape[0]),
        replace=False
    )
    background = X_train_dense[bg_idx]

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional"
    )
    try:
        sv = explainer.shap_values(X_val_dense)
        return sv[1] if isinstance(sv, list) else sv
    except Exception:
        return explainer(X_val_dense).values

def linear_shap(model, X_train_dense: np.ndarray, X_val_dense: np.ndarray):
    # For LR, LinearExplainer is appropriate and fast
    explainer = shap.LinearExplainer(model, X_train_dense)
    try:
        sv = explainer.shap_values(X_val_dense)
        return sv  # already (n, d)
    except Exception:
        return explainer(X_val_dense).values

# ---------- main ----------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    log(f"Working directory: {base.resolve()}")

    # Artifacts
    prep_path = base / "preprocessor.joblib"
    xgb_path  = base / "model_xgb.joblib"
    lr_path   = base / "model_logreg.joblib"
    if not prep_path.exists():
        raise FileNotFoundError("preprocessor.joblib not found.")
    preprocessor = joblib.load(prep_path)

    # Data (baseline-consistent)
    X_train, y_train, X_val, y_val = rebuild_splits(base)

    Xtr = to_dense(preprocessor, X_train)
    Xvl = to_dense(preprocessor, X_val)


    # Feature names (expanded, human-readable)
    feature_names = get_feature_names_from_ct(preprocessor, X_train)
    if len(feature_names) != Xvl.shape[1]:
        feature_names = [f"f{i}" for i in range(Xvl.shape[1])]
    Xvl_df = pd.DataFrame(Xvl, columns=feature_names)

    # Group map to original columns (for grouped bar)
    groups = build_original_groups(preprocessor, X_train, feature_names)

    # ---- XGBoost ----
    if xgb_path.exists():
        log("Generating global SHAP for XGBoost...")
        xgb = joblib.load(xgb_path)
        shap_vals = tree_shap(xgb, Xtr, Xvl)
        plot_beeswarm_and_bar(
            shap_vals, Xvl_df,
            beeswarm_path=base / "xgb_shap_beeswarm.png",
            bar_path=base / "xgb_shap_bar.png",
        )
        plot_grouped_bar(shap_vals, groups, base / "xgb_shap_bar_grouped.png")
    else:
        log("model_xgb.joblib not found; skipping XGBoost.")

    # ---- Logistic Regression ----
    if lr_path.exists():
        log("Generating global SHAP for Logistic Regression...")
        lr = joblib.load(lr_path)
        shap_vals = linear_shap(lr, Xtr, Xvl)
        plot_beeswarm_and_bar(
            shap_vals, Xvl_df,
            beeswarm_path=base / "lr_shap_beeswarm.png",
            bar_path=base / "lr_shap_bar.png",
        )
        plot_grouped_bar(shap_vals, groups, base / "lr_shap_bar_grouped.png")
    else:
        log("model_logreg.joblib not found; skipping Logistic Regression.")

    log("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
