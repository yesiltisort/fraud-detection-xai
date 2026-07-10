# -*- coding: utf-8 -*-


import os, re, sys, json, warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import issparse

import shap
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ===== Baseline-compatible split settings =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000
ANON_PATTERNS = [r"^V\d+$", r"^C\d+$", r"^D\d+$", r"^M\d+$", r"^id_\d+$"]

# ===== Local SHAP config =====
N_SAMPLES = 500   
TOP_K     = 10    


def log(msg: str):
    print(f"[info] {msg}")


# ---------- Data helpers ----------
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

    # Optional downsample to mirror baseline
    if SUBSET_SIZE < len(df):
        keep_frac = SUBSET_SIZE / len(df)
        sss_keep = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1.0 - keep_frac),
            random_state=RANDOM_STATE,
        )
        keep_idx, _ = next(sss_keep.split(X_full, y_full))
        df = df.iloc[keep_idx].reset_index(drop=True)
        log(f"Downsampled to {len(df)} rows.")

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    # gentle numeric downcast
    X = X.apply(pd.to_numeric, errors="ignore", downcast="float")
    X = X.apply(pd.to_numeric, errors="ignore", downcast="integer")

    # Outer split: train+val vs test
    sss_outer = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    # Inner split: train vs val
    sss_inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_FRACTION_OF_TEMP,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def to_dense(preprocessor, X: pd.DataFrame) -> np.ndarray:
    Xt = preprocessor.transform(X)
    return Xt.toarray() if issparse(Xt) else Xt


# ---------- Feature names extraction ----------
def get_feature_names_from_ct(preprocessor, input_df: pd.DataFrame) -> List[str]:
    
    # Fast path (sklearn >= 1.0)
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        pass

    names: List[str] = []
    used_cols = []

    if isinstance(preprocessor, ColumnTransformer):
        transformers = preprocessor.transformers_
    elif isinstance(preprocessor, Pipeline):
        
        last_step = preprocessor.steps[-1][1]
        if isinstance(last_step, ColumnTransformer):
            transformers = last_step.transformers_
        else:
            transformers = []
    else:
        transformers = []

    for name, trans, cols in transformers:
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
                cats_list = final_step.categories_
                for i, col in enumerate(cols):
                    cats = cats_list[i]
                    names.extend([f"{col}__{str(cat)}" for cat in cats])
            else:
                feat_names = input_df.columns[cols].tolist()
                cats_list = final_step.categories_
                for i, col in enumerate(feat_names):
                    cats = cats_list[i]
                    names.extend([f"{col}__{str(cat)}" for cat in cats])
        else:
            if isinstance(cols, (list, tuple)):
                names.extend(list(cols))
            else:
                names.extend(input_df.columns[cols].tolist())

    # final length-safe fallback
    dim = to_dense(preprocessor, input_df.iloc[:1]).shape[1]
    if len(names) != dim:
        names = [f"f{i}" for i in range(dim)]
    return names


# ---------- SHAP utilities ----------
def make_tree_explainer(model, X_train_dense: np.ndarray, seed: int = RANDOM_STATE):
    """Build a TreeExplainer with a small background set for XGBoost."""
    rng = np.random.default_rng(seed)
    bg_size = min(1000, X_train_dense.shape[0])
    bg_idx = rng.choice(X_train_dense.shape[0], size=bg_size, replace=False)
    background = X_train_dense[bg_idx]

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )

    base_value = None
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            base_value = float(ev[1])  # class 1
        else:
            base_value = float(ev)
    except Exception:
        base_value = None

    return explainer, base_value, bg_size


def make_linear_explainer(model, X_train_dense: np.ndarray):
    explainer = shap.LinearExplainer(model, X_train_dense)
    base_value = None
    try:
        base_value = float(explainer.expected_value)
    except Exception:
        base_value = None
    return explainer, base_value


def shap_values_class1(explainer, X_dense: np.ndarray) -> np.ndarray:
    
    try:
        sv = explainer.shap_values(X_dense)  # old API
        if isinstance(sv, list):
            return sv[1]  # pick class 1
        return sv
    except Exception:
        exp = explainer(X_dense)
        return exp.values  # (n, d)


# ---------- Main pipeline ----------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    log(f"Working directory: {base.resolve()}")

    # Artifacts
    prep_path = base / "preprocessor.joblib"
    xgb_path = base / "model_xgb.joblib"
    lr_path = base / "model_logreg.joblib"

    if not prep_path.exists():
        raise FileNotFoundError("preprocessor.joblib not found.")
    preprocessor = joblib.load(prep_path)

    # Data
    X_train, y_train, X_val, y_val = rebuild_splits(base)
    Xtr = to_dense(preprocessor, X_train)
    Xvl = to_dense(preprocessor, X_val)

    # Feature names (expanded and semantic)
    feature_names = get_feature_names_from_ct(preprocessor, X_train)
    if len(feature_names) != Xvl.shape[1]:
        feature_names = [f"f{i}" for i in range(Xvl.shape[1])]

    # Choose subset to explain
    rng = np.random.default_rng(RANDOM_STATE)
    n = Xvl.shape[0]
    sel_idx = rng.choice(n, size=min(N_SAMPLES, n), replace=False)
    Xvl_sel = Xvl[sel_idx]

    # Helper: save feature names
    def save_feature_names(prefix: str):
        p = base / f"{prefix}_feature_names.txt"
        with open(p, "w", encoding="utf-8") as f:
            for nm in feature_names:
                f.write(str(nm) + "\n")
        log(f"Saved: {p.resolve()}")

    # ===== XGBoost =====
    if xgb_path.exists():
        log("Explaining XGBoost locally with SHAP.")
        xgb = joblib.load(xgb_path)
        explainer, base_value_xgb, bg_size = make_tree_explainer(
            xgb, Xtr, seed=RANDOM_STATE
        )
        shap_vals = shap_values_class1(explainer, Xvl_sel)  # (m, d)
        p1 = xgb.predict_proba(Xvl_sel)[:, 1]

        # Store dense SHAP matrix
        npy_path = base / "xgb_shap_contrib_matrix.npy"
        np.save(npy_path, shap_vals)
        log(f"Saved: {npy_path.resolve()}  (shape={shap_vals.shape})")

        # Save feature names and meta
        save_feature_names("xgb")
        meta = {
            "base_value": base_value_xgb,
            "background_size": int(bg_size),
            "random_state": RANDOM_STATE,
            "n_samples": int(shap_vals.shape[0]),
            "n_features": int(shap_vals.shape[1]),
            "note": "SHAP values correspond to class=1 (fraud).",
        }
        with open(base / "xgb_shap_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Build top-k CSV
        rows = []
        for r in range(shap_vals.shape[0]):
            sv = shap_vals[r]
            top_idx = np.argsort(np.abs(sv))[::-1][:TOP_K]
            rows.append(
                {
                    "row_id": int(sel_idx[r]),
                    "pred_proba": float(p1[r]),
                    "topk_indices": "|".join(map(str, top_idx.tolist())),
                    "topk_names": "|".join([feature_names[j] for j in top_idx]),
                    "topk_shap_values": "|".join(
                        [f"{sv[j]:.6f}" for j in top_idx]
                    ),
                }
            )
        out_csv = base / "xgb_shap_local_topk.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log(f"Saved: {out_csv.resolve()}  (N={len(rows)}, top_k={TOP_K})")
    else:
        log("model_xgb.joblib not found; skipping XGBoost.")

    # ===== Logistic Regression =====
    if lr_path.exists():
        log("Explaining Logistic Regression locally with SHAP.")
        lr = joblib.load(lr_path)
        explainer_lr, base_value_lr = make_linear_explainer(lr, Xtr)
        shap_vals_lr = shap_values_class1(explainer_lr, Xvl_sel)  # (m, d)
        p1_lr = lr.predict_proba(Xvl_sel)[:, 1]

        npy_path_lr = base / "lr_shap_contrib_matrix.npy"
        np.save(npy_path_lr, shap_vals_lr)
        log(f"Saved: {npy_path_lr.resolve()}  (shape={shap_vals_lr.shape})")

        save_feature_names("lr")
        meta_lr = {
            "base_value": base_value_lr,
            "random_state": RANDOM_STATE,
            "n_samples": int(shap_vals_lr.shape[0]),
            "n_features": int(shap_vals_lr.shape[1]),
            "note": "SHAP values correspond to class=1 (fraud).",
        }
        with open(base / "lr_shap_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_lr, f, ensure_ascii=False, indent=2)

        rows_lr = []
        for r in range(shap_vals_lr.shape[0]):
            sv = shap_vals_lr[r]
            top_idx = np.argsort(np.abs(sv))[::-1][:TOP_K]
            rows_lr.append(
                {
                    "row_id": int(sel_idx[r]),
                    "pred_proba": float(p1_lr[r]),
                    "topk_indices": "|".join(map(str, top_idx.tolist())),
                    "topk_names": "|".join([feature_names[j] for j in top_idx]),
                    "topk_shap_values": "|".join(
                        [f"{sv[j]:.6f}" for j in top_idx]
                    ),
                }
            )
        out_csv_lr = base / "lr_shap_local_topk.csv"
        pd.DataFrame(rows_lr).to_csv(out_csv_lr, index=False)
        log(f"Saved: {out_csv_lr.resolve()}  (N={len(rows_lr)}, top_k={TOP_K})")
    else:
        log("model_logreg.joblib not found; skipping Logistic Regression.")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
