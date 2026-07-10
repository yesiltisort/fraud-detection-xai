# -*- coding: utf-8 -*-


import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import joblib

from preprocess_baseline2 import load_preprocessed_data

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
BOOTSTRAPS = 50       
PAIR_SAMPLES = 300    

META_COLS = {
    "row_id",
    "isFraud",
    "query_index",
    "LR_pred_proba",
    "LR_cf_pred_proba",
    "XGB_pred_proba",
    "XGB_cf_pred_proba",
}

def log(msg: str) -> None:
    print(f"[info] {msg}")


# ------------- HELPERS -------------
def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def build_baseline_and_feature_names() -> (np.ndarray, List[str]):

    X_train, X_test, y_train, y_test = load_preprocessed_data()

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    n_features = X_train.shape[1]
    feature_names = [f"f{i}" for i in range(n_features)]

    # Eğer lr_feature_names.txt varsa, kullan
    base = Path.cwd()
    lr_names_path = base / "lr_feature_names.txt"
    if lr_names_path.exists():
        with open(lr_names_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == n_features:
            feature_names = names
            X_train.columns = feature_names
            log("Loaded feature names from lr_feature_names.txt")
        else:
            X_train.columns = feature_names
            log("[warn] lr_feature_names.txt length mismatch, using f0..fN")
    else:
        X_train.columns = feature_names
        log("[warn] lr_feature_names.txt not found, using f0..fN")

    baseline_vec = X_train[feature_names].mean(axis=0).values.astype(float)
    log(f"Baseline vector computed from {X_train.shape[0]} training rows.")
    return baseline_vec, feature_names


def predict_proba_pos(model, X: np.ndarray) -> np.ndarray:
    """
    P(y=1) döner.
    """
    return model.predict_proba(X)[:, 1]


# --------- METRIC 
def compute_dice_metrics_lr() -> pd.DataFrame:
    base = Path.cwd()

    lr_path = base / "model_logreg.joblib"
    if not lr_path.exists():
        raise FileNotFoundError("model_logreg.joblib bulunamadı.")

    lr_model = joblib.load(lr_path)
    log("Loaded Logistic Regression model.")


    baseline_vec, feature_names = build_baseline_and_feature_names()


    q_csv = base / "dice_results_LR_dice_queries.csv"
    cf_csv = base / "dice_results_LR_dice_cfs.csv"

    if not q_csv.exists():
        raise FileNotFoundError("dice_results_LR_dice_queries.csv bulunamadı.")
    if not cf_csv.exists():
        raise FileNotFoundError("dice_results_LR_dice_cfs.csv bulunamadı.")

    q_df = pd.read_csv(q_csv)
    cf_df = pd.read_csv(cf_csv)

    if "row_id" not in q_df.columns or "row_id" not in cf_df.columns:
        raise ValueError("LR DiCE CSV'lerinde 'row_id' kolonu yok.")


    feature_cols = [
        c
        for c in q_df.columns
        if (
            c not in META_COLS
            and not c.endswith("_pred_proba")
            and not c.endswith("_cf_pred_proba")
        )
    ]

    feature_cols = [c for c in feature_names if c in feature_cols]

    if not feature_cols:
        raise ValueError("Feature kolonları bulunamadı, metrik hesaplanamıyor.")

    log(f"LR: using {len(feature_cols)} features for metrics.")

    q_by_id = {int(r["row_id"]): r for _, r in q_df.iterrows()}
    cf_grouped = cf_df.groupby("row_id")

    common_ids = sorted(set(q_by_id.keys()) & set(cf_grouped.groups.keys()))
    if not common_ids:
        raise ValueError("LR için ortak row_id bulunamadı (queries vs cfs).")

    baseline = baseline_vec.astype(float)

    comp_list = []
    suf_list = []
    sets_for_stability: List[Set[int]] = []

    for rid in common_ids:
        q_row = q_by_id[rid]
        group = cf_grouped.get_group(rid)
        cf_row = group.iloc[0]  

        x_orig = np.asarray(q_row[feature_cols].values, dtype=float)
        x_cf = np.asarray(cf_row[feature_cols].values, dtype=float)

        
        changed_idx = np.where(np.abs(x_cf - x_orig) > 1e-8)[0]
        if changed_idx.size == 0:
            continue


        p_orig = float(predict_proba_pos(lr_model, x_orig.reshape(1, -1))[0])


        x_removed = x_orig.copy()
        x_removed[changed_idx] = baseline[changed_idx]
        p_removed = float(predict_proba_pos(lr_model, x_removed.reshape(1, -1))[0])
        comp = p_orig - p_removed


        x_kept = baseline.copy()
        x_kept[changed_idx] = x_orig[changed_idx]
        p_kept = float(predict_proba_pos(lr_model, x_kept.reshape(1, -1))[0])
        suf = p_orig - p_kept

        comp_list.append(comp)
        suf_list.append(suf)
        sets_for_stability.append(set(changed_idx.tolist()))

    if not comp_list:
        raise ValueError("LR için geçerli (değişen feature'lı) örnek bulunamadı.")

    mean_comp = float(np.mean(comp_list))
    std_comp = float(np.std(comp_list))
    mean_suf = float(np.mean(suf_list))
    std_suf = float(np.std(suf_list))


    if len(sets_for_stability) < 2:
        stability_mean = float("nan")
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        vals = []
        for _ in range(BOOTSTRAPS):
            js = []
            for _ in range(PAIR_SAMPLES):
                a_idx, b_idx = rng.choice(len(sets_for_stability), size=2, replace=False)
                js.append(jaccard(sets_for_stability[a_idx], sets_for_stability[b_idx]))
            vals.append(np.mean(js))
        stability_mean = float(np.mean(vals))

    rows = [
        {
            "model": "LR",
            "metric": "comprehensiveness",
            "k": -1,
            "mean": mean_comp,
            "std": std_comp,
        },
        {
            "model": "LR",
            "metric": "sufficiency",
            "k": -1,
            "mean": mean_suf,
            "std": std_suf,
        },
        {
            "model": "LR",
            "metric": "stability_jaccard",
            "k": -1,
            "mean": stability_mean,
            "std": np.nan,
        },
    ]

    return pd.DataFrame(rows)


# -------------- MAIN --------------
def main():
    try:
        df = compute_dice_metrics_lr()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)

    base = Path.cwd()
    out_csv = base / "dice_metrics_lr_summary.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved: {out_csv.resolve()}")

    pivot = df.pivot_table(index="model", columns="metric", values="mean")
    print("\n=== DICE METRICS (LR ONLY) ===")
    print(pivot.round(4))


if __name__ == "__main__":
    main()
