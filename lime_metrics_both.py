# -*- coding: utf-8 -*-

import os, re, sys, warnings, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import issparse

# ===== baseline split settings (match your pipeline) =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000
ANON_PATTERNS = [r"^V\\d+$", r"^C\\d+$", r"^D\\d+$", r"^M\\d+$", r"^id_\\d+$"]

# ===== metric params =====
K_VALUES = [5, 10, 20]
BOOTSTRAPS = 100        
PAIR_SAMPLES = 1000     


def log(m: str) -> None:
    print(f"[info] {m}")


# ---------- data & split helpers ----------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.replace("-", "_") for c in out.columns]
    return out


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

    # Optional subsampling to speed up the metrics
    if SUBSET_SIZE < len(df):
        keep_frac = SUBSET_SIZE / len(df)
        sss_keep = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1.0 - keep_frac),
            random_state=RANDOM_STATE,
        )
        keep_idx, _ = next(sss_keep.split(X_full, y_full))
        df = df.iloc[keep_idx].reset_index(drop=True)

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    # light numeric downcasting (optional)
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

    # Inner split: train vs validation
    sss_inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_FRACTION_OF_TEMP,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    return (
        X_train,
        y_train.reset_index(drop=True),
        X_val,
        y_val.reset_index(drop=True),
    )


def to_dense(preprocessor, X: pd.DataFrame) -> np.ndarray:
    Xt = preprocessor.transform(X)
    return Xt.toarray() if issparse(Xt) else Xt


# ---------- metrics helpers ----------
def predict_proba_pos(model, X_dense: np.ndarray) -> np.ndarray:
    
    return model.predict_proba(X_dense)[:, 1]


def jaccard(a: set, b: set) -> float:
  
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / (union if union > 0 else 1.0)


def parse_topk_indices_list(s: str) -> List[int]:
  
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split("|") if str(x).strip().isdigit()]


def compute_sufficiency_comprehensiveness(
    model_name: str,
    model,
    Xtr_dense: np.ndarray,
    Xvl_dense: np.ndarray,
    topk_csv: Path,
    ks: List[int],
) -> pd.DataFrame:
    
    if not topk_csv.exists():
        log(f"{model_name}: {topk_csv.name} not found; skipping S&C.")
        return pd.DataFrame()

    df = pd.read_csv(topk_csv)

    # Optional: Restrict to the same evaluation indices used by SHAP
 
    base = Path.cwd()
    shap_idx_path = None
    if model_name.lower() == "xgb":
        shap_idx_path = base / "xgb_shap_eval_indices.npy"
    elif model_name.lower() == "lr":
        shap_idx_path = base / "lr_shap_eval_indices.npy"

    if shap_idx_path is not None and shap_idx_path.exists():
        shap_indices = np.load(shap_idx_path)
        shap_indices = set(int(i) for i in shap_indices)
        df = df[df["row_id"].isin(shap_indices)].reset_index(drop=True)
        log(f"{model_name}: using {len(df)} rows that overlap with {shap_idx_path.name}")
    else:
        log(f"{model_name}: no SHAP eval index file found; using all rows from {topk_csv.name}")

    # Row ids were stored as indices of the validation array in the LIME script
    row_ids = df["row_id"].astype(int).tolist()

    # Baseline vector (training mean in transformed space)
    baseline = Xtr_dense.mean(axis=0)

    # Only evaluate rows that exist in the validation matrix
    row_ids = [rid for rid in row_ids if 0 <= rid < Xvl_dense.shape[0]]
    if len(row_ids) == 0:
        log(f"{model_name}: no valid row ids found in {topk_csv.name}")
        return pd.DataFrame()

    # Precompute original probabilities
    p_orig = predict_proba_pos(model, Xvl_dense[row_ids])

    out_rows = []

    # Pre-parse per-instance top-k list (full set from CSV, we will slice to k)
    all_pairs = []
    for _, r in df.iterrows():
        rid = int(r["row_id"])
        idxs = parse_topk_indices_list(r.get("topk_indices", ""))
        all_pairs.append((rid, idxs))

    # Build quick map: row_id -> list of indices
    idx_map = {rid: idxs for rid, idxs in all_pairs}

    for k in ks:
        comps, sufs = [], []

        for i, rid in enumerate(row_ids):
            x = Xvl_dense[rid].copy()
            topk = idx_map.get(rid, [])
            if not topk:
                continue

            topk = topk[:k] if len(topk) > k else topk

            # removed: set top-k features to baseline
            x_removed = x.copy()
            x_removed[topk] = baseline[topk]
            p_removed = predict_proba_pos(model, x_removed.reshape(1, -1))[0]

            # kept: start from baseline, then keep only top-k features from x
            x_kept = baseline.copy()
            x_kept[topk] = x[topk]
            p_kept = predict_proba_pos(model, x_kept.reshape(1, -1))[0]

            comp = p_orig[i] - p_removed      # higher is better
            suf = p_orig[i] - p_kept          # lower is better
            comps.append(comp)
            sufs.append(suf)

        if len(comps) == 0:
            mean_c, std_c, mean_s, std_s = np.nan, np.nan, np.nan, np.nan
        else:
            mean_c, std_c = float(np.mean(comps)), float(np.std(comps))
            mean_s, std_s = float(np.mean(sufs)), float(np.std(sufs))

        out_rows.append({
            "model": model_name.upper(),
            "k": k,
            "metric": "comprehensiveness",
            "mean": mean_c,
            "std": std_c,
        })
        out_rows.append({
            "model": model_name.upper(),
            "k": k,
            "metric": "sufficiency",
            "mean": mean_s,
            "std": std_s,
        })

    return pd.DataFrame(out_rows)


def compute_stability(
    model_name: str,
    topk_csv: Path,
    k: int,
    bootstraps: int = BOOTSTRAPS,
    pair_samples: int = PAIR_SAMPLES,
) -> float:

    if not topk_csv.exists():
        log(f"{model_name}: {topk_csv.name} not found; stability skipped.")
        return float("nan")

    df = pd.read_csv(topk_csv)


    base = Path.cwd()
    shap_idx_path = None
    if model_name.lower() == "xgb":
        shap_idx_path = base / "xgb_shap_eval_indices.npy"
    elif model_name.lower() == "lr":
        shap_idx_path = base / "lr_shap_eval_indices.npy"

    if shap_idx_path is not None and shap_idx_path.exists():
        shap_indices = np.load(shap_idx_path)
        shap_indices = set(int(i) for i in shap_indices)
        df = df[df["row_id"].isin(shap_indices)].reset_index(drop=True)
        log(f"{model_name}: stability uses {len(df)} rows overlapping with {shap_idx_path.name}")
    else:
        log(f"{model_name}: stability uses all rows from {topk_csv.name}")

    sets = []
    for _, r in df.iterrows():
        idxs = parse_topk_indices_list(r.get("topk_indices", ""))
        if not idxs:
            continue
        sets.append(set(idxs[:k]))

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


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    base = Path.cwd()

    # Artifacts
    prep_path = base / "preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError("preprocessor.joblib not found.")
    preprocessor = joblib.load(prep_path)

    # Data
    X_train, y_train, X_val, y_val = rebuild_splits(base)
    Xtr = to_dense(preprocessor, X_train)
    Xvl = to_dense(preprocessor, X_val)

    rows = []

    # ---- XGBoost ----
    xgb_model_path = base / "model_xgb.joblib"
    xgb_topk_csv = base / "xgb_lime_local_topk.csv"
    if xgb_model_path.exists() and xgb_topk_csv.exists():
        xgb = joblib.load(xgb_model_path)
        log("Computing LIME metrics for XGBoost...")
        df_sc = compute_sufficiency_comprehensiveness(
            model_name="xgb",
            model=xgb,
            Xtr_dense=Xtr,
            Xvl_dense=Xvl,
            topk_csv=xgb_topk_csv,
            ks=K_VALUES,
        )
        # Stability per k
        stab_rows = []
        for k in K_VALUES:
            s = compute_stability(
                "xgb",
                xgb_topk_csv,
                k=k,
                bootstraps=BOOTSTRAPS,
                pair_samples=PAIR_SAMPLES,
            )
            stab_rows.append({
                "model": "XGB",
                "k": k,
                "metric": "stability_jaccard",
                "mean": s,
                "std": np.nan,
            })
        rows.append(df_sc)
        rows.append(pd.DataFrame(stab_rows))
    else:
        log("Skipping XGBoost (missing model or LIME CSV).")

    # ---- Logistic Regression ----
    lr_model_path = base / "model_logreg.joblib"
    lr_topk_csv = base / "lr_lime_local_topk.csv"
    if lr_model_path.exists() and lr_topk_csv.exists():
        lr = joblib.load(lr_model_path)
        log("Computing LIME metrics for Logistic Regression...")
        df_sc = compute_sufficiency_comprehensiveness(
            model_name="lr",
            model=lr,
            Xtr_dense=Xtr,
            Xvl_dense=Xvl,
            topk_csv=lr_topk_csv,
            ks=K_VALUES,
        )
        stab_rows = []
        for k in K_VALUES:
            s = compute_stability(
                "lr",
                lr_topk_csv,
                k=k,
                bootstraps=BOOTSTRAPS,
                pair_samples=PAIR_SAMPLES,
            )
            stab_rows.append({
                "model": "LR",
                "k": k,
                "metric": "stability_jaccard",
                "mean": s,
                "std": np.nan,
            })
        rows.append(df_sc)
        rows.append(pd.DataFrame(stab_rows))
    else:
        log("Skipping Logistic Regression (missing model or LIME CSV).")

    if not rows:
        log("No metrics computed. Ensure LIME outputs and models exist.")
        return

    summary = pd.concat(rows, ignore_index=True)

    # Order rows nicely
    metric_order = {"comprehensiveness": 0, "sufficiency": 1, "stability_jaccard": 2}
    summary["metric_order"] = summary["metric"].map(metric_order).fillna(99)
    summary = summary.sort_values(["model", "metric_order", "k"]).drop(columns=["metric_order"])

    # Save + print
    out_csv = base / "lime_metrics_summary.csv"
    summary.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv.resolve()}")

    # Pretty print
    pivot = summary.pivot_table(index=["model", "k"], columns="metric", values="mean")
    print("\n=== LIME METRICS SUMMARY (mean values) ===")
    print(pivot.round(4))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
