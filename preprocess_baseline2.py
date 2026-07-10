import json
import time
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from scipy.sparse import issparse


try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception as e:
    print("[warn] Could not import XGBoost:", e)
    HAS_XGB = False


# ============================================================
# Global constants
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000

ANON_PATTERNS = [
    r"^V\d+$",
    r"^C\d+$",
    r"^D\d+$",
    r"^M\d+$",
    r"^id_\d+$",
]


def log(msg: str):
    print(f"[info] {msg}")


# ============================================================
# Utility functions
# ============================================================
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def drop_anonymized_cols(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    to_drop = [c for c in df.columns if any(re.match(p, c) for p in ANON_PATTERNS)]
    log(f"Dropping {len(to_drop)} anonymized columns.")
    return df.drop(columns=to_drop, errors="ignore")


def downcast_features(X: pd.DataFrame) -> pd.DataFrame:

    X = X.apply(pd.to_numeric, errors="ignore", downcast="float")
    X = X.apply(pd.to_numeric, errors="ignore", downcast="integer")
    return X


def infer_feature_types(Xdf: pd.DataFrame):
    
    num_cols, cat_cols = [], []
    for col in Xdf.columns:
        dt = Xdf[col].dtype
        if is_numeric_dtype(dt):
            num_cols.append(col)
        elif is_bool_dtype(dt) or is_categorical_dtype(dt) or dt == "object":
            cat_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def make_splits(y: pd.Series):


    y = y.astype(int).reset_index(drop=True)
    n = len(y)
    dummy = np.zeros((n, 1))

    # Outer split: train+val vs test
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trainval_idx, test_idx = next(sss_outer.split(dummy, y))

    # Inner split: train vs val, done within train+val
    y_trainval = y.iloc[trainval_idx]
    dummy_tv = np.zeros((len(trainval_idx), 1))

    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION_OF_TEMP, random_state=RANDOM_STATE
    )
    inner_train_idx, inner_val_idx = next(sss_inner.split(dummy_tv, y_trainval))

    train_idx = trainval_idx[inner_train_idx]
    val_idx = trainval_idx[inner_val_idx]

    return train_idx, val_idx, test_idx


def build_preprocessor_and_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx,
    val_idx,
    test_idx,
):


    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    num_cols, cat_cols = infer_feature_types(X_train)

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    log("Fitting preprocessor...")
    preprocessor.fit(X_train, y_train)

    X_train_tr = preprocessor.transform(X_train)
    X_val_tr = preprocessor.transform(X_val)
    X_test_tr = preprocessor.transform(X_test)

    # Ensure CSR for XGBoost / LogReg
    if issparse(X_train_tr):
        X_train_tr = X_train_tr.tocsr()
    if issparse(X_val_tr):
        X_val_tr = X_val_tr.tocsr()
    if issparse(X_test_tr):
        X_test_tr = X_test_tr.tocsr()

    return preprocessor, X_train_tr, X_val_tr, X_test_tr, y_train, y_val, y_test


def select_threshold_by_f1(y_true, proba):
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-9, None)
    best_idx = np.nanargmax(f1s[:-1])  # ignore last threshold
    return float(thr[best_idx]), float(f1s[:-1][best_idx])


def eval_metrics_for_table(y_val, prob_val, y_test, prob_test, threshold=0.5):
    val_pred = (prob_val >= threshold).astype(int)
    test_pred = (prob_test >= threshold).astype(int)

    return {
        "precision_val": precision_score(y_val, val_pred),
        "recall_val": recall_score(y_val, val_pred),
        "f1_val": f1_score(y_val, val_pred),
        "pr_auc_val": average_precision_score(y_val, prob_val),
        "precision_test": precision_score(y_test, test_pred),
        "recall_test": recall_score(y_test, test_pred),
        "f1_test": f1_score(y_test, test_pred),
        "pr_auc_test": average_precision_score(y_test, prob_test),
    }


def build_results_table(log_res: dict, xgb_res: dict) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "PR-AUC"],
            "Logistic Regression (Val)": [
                log_res["precision_val"],
                log_res["recall_val"],
                log_res["f1_val"],
                log_res["pr_auc_val"],
            ],
            "Logistic Regression (Test)": [
                log_res["precision_test"],
                log_res["recall_test"],
                log_res["f1_test"],
                log_res["pr_auc_test"],
            ],
            "XGBoost (Val)": [
                xgb_res["precision_val"],
                xgb_res["recall_val"],
                xgb_res["f1_val"],
                xgb_res["pr_auc_val"],
            ],
            "XGBoost (Test)": [
                xgb_res["precision_test"],
                xgb_res["recall_test"],
                xgb_res["f1_test"],
                xgb_res["pr_auc_test"],
            ],
        }
    )
    return table


def train_logreg(X_train_tr, y_train, X_val_tr, y_val):


    C_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best = None

    for C in C_grid:
        log(
            f"LogReg: trying C={C}"
        )
        model = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=C,
            max_iter=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        t0 = time.time()
        model.fit(X_train_tr, y_train)
        fit_time = time.time() - t0

        val_prob = model.predict_proba(X_val_tr)[:, 1]
        ap = average_precision_score(y_val, val_prob)

        log(f"  -> C={C} | val AP={ap:.4f} | fit time={fit_time:.2f}s")

        if (best is None) or (ap > best["ap"]):
            best = {"ap": ap, "C": C, "model": model, "val_prob": val_prob}

    best_model = best["model"]
    best_C = best["C"]
    best_ap = best["ap"]
    log(f"Best LogReg C={best_C} with val AP={best_ap:.4f}")


    thr_lr, f1_val = select_threshold_by_f1(y_val, best["val_prob"])
    log(f"LogReg chosen threshold={thr_lr:.4f} (val F1={f1_val:.4f})")

    return best_model, thr_lr, best_C


def train_xgb_optimized(X_train_tr, y_train, X_val_tr, y_val):


    if not HAS_XGB:
        raise RuntimeError("XGBoost is not available.")

    # Handle imbalance
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(1, n_pos)
    log(f"scale_pos_weight ≈ {spw:.2f} (neg={n_neg}, pos={n_pos})")

    param_grid = [
        dict(max_depth=d, min_child_weight=mcw, subsample=sub, colsample_bytree=col)
        for d in [3, 4, 5]
        for mcw in [1, 2, 5]
        for sub in [0.8, 1.0]
        for col in [0.6, 0.8, 1.0]
    ]

    best = None
    for i, p in enumerate(param_grid, 1):
        log(
            f"XGB param set {i}/{len(param_grid)}: "
            f"max_depth={p['max_depth']}, min_child_weight={p['min_child_weight']}, "
            f"subsample={p['subsample']}, colsample_bytree={p['colsample_bytree']}"
        )

        model = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=p["max_depth"],
            min_child_weight=p["min_child_weight"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            scale_pos_weight=spw,
            n_jobs=-1,
        )

        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            verbose=False,
            early_stopping_rounds=100,
        )

        val_prob = model.predict_proba(X_val_tr)[:, 1]
        ap = average_precision_score(y_val, val_prob)
        auc = roc_auc_score(y_val, val_prob)
        score = ap

        if (best is None) or (score > best["score"]):
            best = {
                "score": score,
                "auc": auc,
                "params": p,
                "model": model,
                "best_ntree": getattr(model, "best_iteration", None),
            }

    assert best is not None
    log(
        f"Best XGB params: {best['params']} | "
        f"Val AP={best['score']:.4f}, AUC={best['auc']:.4f}, "
        f"best_ntree={best['best_ntree']}"
    )

    # Threshold selection on validation set (maximize F1)
    best_val_prob = best["model"].predict_proba(X_val_tr)[:, 1]
    thr, f1v = select_threshold_by_f1(y_val, best_val_prob)
    log(f"Chosen threshold (max F1 on val): {thr:.4f} | Val F1={f1v:.4f}")

    return best["model"], thr, best["params"], spw


# ============================================================
# Main
# ============================================================
def main():
    base = Path.cwd()
    log(f"Working directory: {base.resolve()}")

    # -------------------------------
    # 1. Read and merge raw data
    # -------------------------------
    tx = pd.read_csv(base / "train_transaction.csv")
    idf = pd.read_csv(base / "train_identity.csv")
    df = tx.merge(idf, on="TransactionID", how="left")
    df = standardize_cols(df)

    if "isFraud" not in df.columns:
        raise RuntimeError("Target column 'isFraud' not found!")

    # -------------------------------
    # 2. Stratified downsampling
    # -------------------------------
    y_full = df["isFraud"].astype(int)
    X_full = df.drop(columns=["isFraud"], errors="ignore")
    n_full = len(df)

    if SUBSET_SIZE < n_full:
        keep_frac = SUBSET_SIZE / n_full
        sss_keep = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - keep_frac), random_state=RANDOM_STATE
        )
        keep_idx, _ = next(sss_keep.split(X_full, y_full))
        df = df.iloc[keep_idx].reset_index(drop=True)
        log(f"Downsampled from {n_full} → {len(df)} rows")
    else:
        df = df.reset_index(drop=True)
        log("No downsampling performed.")

    # Common target and splits
    y_all = df["isFraud"].astype(int)
    train_idx, val_idx, test_idx = make_splits(y_all)

    # =======================================================
    # Scenario 1: KEEP anonymized namespaces
    # =======================================================
    log("===== Scenario 1: WITH anonymized columns =====")

    X_with = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    X_with = downcast_features(X_with)

    (
        prep_with,
        Xtr_with,
        Xvl_with,
        Xte_with,
        ytr_with,
        yvl_with,
        yte_with,
    ) = build_preprocessor_and_data(X_with, y_all, train_idx, val_idx, test_idx)

    # Logistic Regression (tuned C + tuned threshold)
    log_reg_with, thr_lr_with, C_with = train_logreg(
        Xtr_with, ytr_with, Xvl_with, yvl_with
    )
    log_val_prob_with = log_reg_with.predict_proba(Xvl_with)[:, 1]
    log_test_prob_with = log_reg_with.predict_proba(Xte_with)[:, 1]
    log_res_with = eval_metrics_for_table(
        yvl_with,
        log_val_prob_with,
        yte_with,
        log_test_prob_with,
        threshold=thr_lr_with,  
    )
    log(f"Scenario 1 LogReg: best C={C_with}, threshold={thr_lr_with:.4f}")


    # XGBoost
    if HAS_XGB:
        xgb_with, thr_with, params_with, spw_with = train_xgb_optimized(
            Xtr_with, ytr_with, Xvl_with, yvl_with
        )
        xgb_val_prob_with = xgb_with.predict_proba(Xvl_with)[:, 1]
        xgb_test_prob_with = xgb_with.predict_proba(Xte_with)[:, 1]
        xgb_res_with = eval_metrics_for_table(
            yvl_with, xgb_val_prob_with, yte_with, xgb_test_prob_with, threshold=thr_with
        )

        table_with = build_results_table(log_res_with, xgb_res_with)
        print("\n=== RESULTS WITH ANONYMIZED COLUMNS ===")
        print(table_with.to_string(index=False))
    else:
        print("\n[warn] XGBoost not available, skipping XGB for Scenario 1.")

    # =======================================================
    # Scenario 2: DROP anonymized namespaces
    # =======================================================
    log("===== Scenario 2: WITHOUT anonymized columns =====")

    df_noanon = drop_anonymized_cols(df).reset_index(drop=True)
    X_noanon = df_noanon.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    X_noanon = downcast_features(X_noanon)

    (
        prep_noanon,
        Xtr_noanon,
        Xvl_noanon,
        Xte_noanon,
        ytr_noanon,
        yvl_noanon,
        yte_noanon,
    ) = build_preprocessor_and_data(X_noanon, y_all, train_idx, val_idx, test_idx)

    # -------------------------------------------------------
    # Save preprocessed NON-anonymized test set to CSV
    # -------------------------------------------------------

    # Convert sparse matrix to dense array if necessary
    if issparse(Xte_noanon):
        X_test_dense = Xte_noanon.toarray()
    else:
        X_test_dense = Xte_noanon

    # Create DataFrame for X_test and save
    X_test_df = pd.DataFrame(X_test_dense)
    X_test_df.to_csv(base / "X_test.csv", index=False)

    # Create DataFrame for y_test and save

    y_test_df = pd.DataFrame({"isFraud": np.asarray(yte_noanon)})
    y_test_df.to_csv(base / "y_test.csv", index=False)

    log("Saved X_test.csv and y_test.csv for the NON-anonymized scenario.")

    # Save train set too (required by load_preprocessed_data)
    X_train_df = pd.DataFrame(Xtr_noanon.toarray() if issparse(Xtr_noanon) else Xtr_noanon)
    X_train_df.to_csv(base / "X_train.csv", index=False)
    
    y_train_df = pd.DataFrame({"isFraud": np.asarray(ytr_noanon)})
    y_train_df.to_csv(base / "y_train.csv", index=False)
    
    log("Saved X_train.csv and y_train.csv for the NON-anonymized scenario.")
    

    # Logistic Regression (tuned C + tuned threshold)
    log_reg_noanon, thr_lr_noanon, C_noanon = train_logreg(
        Xtr_noanon, ytr_noanon, Xvl_noanon, yvl_noanon
    )
    log_val_prob_noanon = log_reg_noanon.predict_proba(Xvl_noanon)[:, 1]
    log_test_prob_noanon = log_reg_noanon.predict_proba(Xte_noanon)[:, 1]
    log_res_noanon = eval_metrics_for_table(
        yvl_noanon,
        log_val_prob_noanon,
        yte_noanon,
        log_test_prob_noanon,
        threshold=thr_lr_noanon,
    )
    log(f"Scenario 2 LogReg: best C={C_noanon}, threshold={thr_lr_noanon:.4f}")
    

    # XGBoost (optimized)
    if HAS_XGB:
        xgb_noanon, thr_noanon, params_noanon, spw_noanon = train_xgb_optimized(
            Xtr_noanon, ytr_noanon, Xvl_noanon, yvl_noanon
        )
        xgb_val_prob_noanon = xgb_noanon.predict_proba(Xvl_noanon)[:, 1]
        xgb_test_prob_noanon = xgb_noanon.predict_proba(Xte_noanon)[:, 1]
        xgb_res_noanon = eval_metrics_for_table(
            yvl_noanon,
            xgb_val_prob_noanon,
            yte_noanon,
            xgb_test_prob_noanon,
            threshold=thr_noanon,
        )

        table_noanon = build_results_table(log_res_noanon, xgb_res_noanon)
        print("\n=== RESULTS WITHOUT ANONYMIZED COLUMNS ===")
        print(table_noanon.to_string(index=False))
    else:
        print("[warn] XGBoost not available, skipping XGB for Scenario 2.")
        xgb_noanon = None

    # =======================================================
    # Save final artifacts (NON-anonymized feature set)
    # =======================================================
    log("Saving final artifacts based on NON-anonymized feature set...")

    joblib.dump(prep_noanon, base / "preprocessor.joblib")
    joblib.dump(log_reg_noanon, base / "model_logreg.joblib")

    if HAS_XGB and xgb_noanon is not None:
        joblib.dump(xgb_noanon, base / "model_xgb.joblib")

        # Build JSON metadata similar to xgb_optimize.py
        val_ap = average_precision_score(yvl_noanon, xgb_val_prob_noanon)
        val_auc = roc_auc_score(yvl_noanon, xgb_val_prob_noanon)
        test_ap = average_precision_score(yte_noanon, xgb_test_prob_noanon)
        test_auc = roc_auc_score(yte_noanon, xgb_test_prob_noanon)

        meta = {
            "best_params": params_noanon,
            "best_iteration": getattr(xgb_noanon, "best_iteration", None),
            "scale_pos_weight": float(spw_noanon),
            "threshold": float(thr_noanon),
            "val_AP": float(val_ap),
            "val_AUC": float(val_auc),
            "val_precision_at_threshold": float(xgb_res_noanon["precision_val"]),
            "val_recall_at_threshold": float(xgb_res_noanon["recall_val"]),
            "val_F1_at_threshold": float(xgb_res_noanon["f1_val"]),
            "test_AP": float(test_ap),
            "test_AUC": float(test_auc),
            "test_precision_at_threshold": float(xgb_res_noanon["precision_test"]),
            "test_recall_at_threshold": float(xgb_res_noanon["recall_test"]),
            "test_F1_at_threshold": float(xgb_res_noanon["f1_test"]),
        }

        with open(base / "xgb_best.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        log("Saved: preprocessor.joblib, model_logreg.joblib, model_xgb.joblib, xgb_best.json")
    else:
        log("Saved: preprocessor.joblib, model_logreg.joblib (no XGBoost model).")

    log("All done.")

def load_preprocessed_data(base_path="."):
  
  
    base = Path(base_path)

    # Load test data
    X_test = pd.read_csv(base / "X_test.csv")
    y_test = pd.read_csv(base / "y_test.csv")["isFraud"]

    # Load train data
    X_train = pd.read_csv(base / "X_train.csv")
    y_train = pd.read_csv(base / "y_train.csv")["isFraud"]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
