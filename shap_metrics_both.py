# -*- coding: utf-8 -*-


import os, re, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import issparse

# ---------------- Config ----------------
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000
ANON_PATTERNS = [r"^V\\d+$", r"^C\\d+$", r"^D\\d+$", r"^M\\d+$", r"^id_\\d+$"]
K_VALUES = [5, 10, 20]
BOOTSTRAPS = 100
PAIR_SAMPLES = 1000

def log(msg): print(f"[info] {msg}")

# ---------------- Helpers ----------------
def standardize_cols(df):
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df

def drop_anonymized_cols(df):
    to_drop = [c for c in df.columns if any(re.match(p, c) for p in ANON_PATTERNS)]
    return df.drop(columns=to_drop, errors="ignore")

def rebuild_splits(base: Path):
    tx = pd.read_csv(base / "train_transaction.csv")
    idf = pd.read_csv(base / "train_identity.csv")
    df = standardize_cols(tx.merge(idf, on="TransactionID", how="left"))
    df = drop_anonymized_cols(df)
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    if SUBSET_SIZE < len(df):
        keep_frac = SUBSET_SIZE / len(df)
        sss_keep = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - keep_frac), random_state=RANDOM_STATE)
        keep_idx, _ = next(sss_keep.split(X, y))
        df = df.iloc[keep_idx].reset_index(drop=True)
        y = df["isFraud"].astype(int)
        X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]

    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION_OF_TEMP, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
    return X_train, y_train, X_val, y_val

def to_dense(preprocessor, X):
    Xt = preprocessor.transform(X)
    return Xt.toarray() if issparse(Xt) else Xt

def predict_proba_pos(model, X):
    return model.predict_proba(X)[:, 1]

def jaccard(a: set, b: set):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

def parse_topk_indices_list(s: str):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split("|") if x.strip().isdigit()]

# ---------------- Metrics ----------------
def compute_sufficiency_comprehensiveness(model_name, model, Xtr_dense, Xvl_dense, topk_csv, ks):
    if not topk_csv.exists():
        log(f"{model_name}: missing {topk_csv.name}")
        return pd.DataFrame()

    df = pd.read_csv(topk_csv)
    baseline = Xtr_dense.mean(axis=0)
    row_ids = df["row_id"].astype(int).tolist()
    row_ids = [rid for rid in row_ids if 0 <= rid < Xvl_dense.shape[0]]

    if len(row_ids) == 0:
        return pd.DataFrame()

    idx_array = np.asarray(row_ids, dtype=int)
    idx_path = topk_csv.with_name(f"{model_name.lower()}_shap_eval_indices.npy")
    np.save(idx_path, idx_array)
    log(f"{model_name}: saved eval indices to {idx_path.name}")

    p_orig = predict_proba_pos(model, Xvl_dense[row_ids])
    all_pairs = {
        int(r["row_id"]): parse_topk_indices_list(r.get("topk_indices", ""))
        for _, r in df.iterrows()
    }

    results = []
    for k in ks:
        comps, sufs = [], []
        for rid in row_ids:
            x = Xvl_dense[rid].copy()
            topk = all_pairs.get(rid, [])[:k]
            if not topk:
                continue

            x_removed = x.copy()
            x_removed[topk] = baseline[topk]
            p_removed = predict_proba_pos(model, x_removed.reshape(1, -1))[0]

            x_kept = baseline.copy()
            x_kept[topk] = x[topk]
            p_kept = predict_proba_pos(model, x_kept.reshape(1, -1))[0]

            comps.append(p_orig[row_ids.index(rid)] - p_removed)
            sufs.append(p_orig[row_ids.index(rid)] - p_kept)

        results.append({
            "model": model_name.upper(),
            "k": k,
            "metric": "comprehensiveness",
            "mean": np.mean(comps),
            "std": np.std(comps)
        })
        results.append({
            "model": model_name.upper(),
            "k": k,
            "metric": "sufficiency",
            "mean": np.mean(sufs),
            "std": np.std(sufs)
        })
    return pd.DataFrame(results)

def compute_stability(model_name, topk_csv, k, bootstraps=100, pair_samples=1000):
    if not topk_csv.exists():
        return float("nan")
    df = pd.read_csv(topk_csv)
    sets = [set(parse_topk_indices_list(r.get("topk_indices", ""))[:k]) for _, r in df.iterrows() if r.get("topk_indices", "")]
    if len(sets) < 2:
        return float("nan")

    rng = np.random.default_rng(RANDOM_STATE)
    vals = []
    for _ in range(bootstraps):
        js = []
        for _ in range(pair_samples):
            a, b = rng.choice(len(sets), size=2, replace=False)
            js.append(jaccard(sets[a], sets[b]))
        vals.append(np.mean(js))
    return float(np.mean(vals))

# ---------------- Main ----------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    base = Path.cwd()
    preprocessor = joblib.load(base / "preprocessor.joblib")
    X_train, y_train, X_val, y_val = rebuild_splits(base)
    Xtr = to_dense(preprocessor, X_train)
    Xvl = to_dense(preprocessor, X_val)

    summary_list = []

    # XGBoost
    if (base / "model_xgb.joblib").exists():
        xgb = joblib.load(base / "model_xgb.joblib")
        log("Evaluating SHAP metrics for XGBoost...")
        xgb_csv = base / "xgb_shap_local_topk.csv"
        df_sc = compute_sufficiency_comprehensiveness("xgb", xgb, Xtr, Xvl, xgb_csv, K_VALUES)
        stab_rows = []
        for k in K_VALUES:
            s = compute_stability("xgb", xgb_csv, k)
            stab_rows.append({"model": "XGB", "k": k, "metric": "stability_jaccard", "mean": s, "std": np.nan})
        summary_list.append(df_sc)
        summary_list.append(pd.DataFrame(stab_rows))

    # Logistic Regression
    if (base / "model_logreg.joblib").exists():
        lr = joblib.load(base / "model_logreg.joblib")
        log("Evaluating SHAP metrics for Logistic Regression...")
        lr_csv = base / "lr_shap_local_topk.csv"
        df_sc = compute_sufficiency_comprehensiveness("lr", lr, Xtr, Xvl, lr_csv, K_VALUES)
        stab_rows = []
        for k in K_VALUES:
            s = compute_stability("lr", lr_csv, k)
            stab_rows.append({"model": "LR", "k": k, "metric": "stability_jaccard", "mean": s, "std": np.nan})
        summary_list.append(df_sc)
        summary_list.append(pd.DataFrame(stab_rows))

    if not summary_list:
        log("No SHAP metrics computed.")
        return

    summary = pd.concat(summary_list, ignore_index=True)
    metric_order = {"comprehensiveness": 0, "sufficiency": 1, "stability_jaccard": 2}
    summary["metric_order"] = summary["metric"].map(metric_order).fillna(99)
    summary = summary.sort_values(["model", "metric_order", "k"]).drop(columns=["metric_order"])

    out_csv = base / "shap_metrics_summary.csv"
    summary.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv.resolve()}")

    pivot = summary.pivot_table(index=["model","k"], columns="metric", values="mean")
    print("\n=== SHAP METRICS SUMMARY (mean values) ===")
    print(pivot.round(4))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
