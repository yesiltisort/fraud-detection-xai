# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import re
import sys
import argparse
import warnings
import inspect
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lime.lime_tabular import LimeTabularExplainer


# ===== Baseline-compatible split settings =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_FRACTION_OF_TEMP = 0.20
SUBSET_SIZE = 100_000  #


ANON_PATTERNS = [r"^V\d+$", r"^C\d+$", r"^D\d+$", r"^M\d+$", r"^id_\d+$"]

# Defaults (can be overridden via CLI)
DEFAULT_TARGET = "isFraud"
DEFAULT_ID = "TransactionID"

TARGET_COL = DEFAULT_TARGET
ID_COL = DEFAULT_ID

# ===== LIME config (improved) =====
N_SAMPLES = 500                 
TOP_K = 20                      
LIME_NUM_FEATURES = 1000      
LIME_NUM_SAMPLES = 12000         
BAG_SEEDS = [7, 23, 41, 57, 83]        


KERNEL_WIDTH_SCALE = {
    "xgb": 0.40,   
    "lr":  1.10,   
}


# --------------------------------------------------------------------------------------
# Utility logging
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[info] {msg}")


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: replace '-' with '_' etc., to keep feature names clean."""
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def drop_anonymized_cols(df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """Drop columns that match anonymized patterns (if you did this in baseline)."""
    if not patterns:
        return df
    to_drop = set()
    for pat in patterns:
        rx = re.compile(pat)
        for c in df.columns:
            if rx.match(c):
                to_drop.add(c)
    if to_drop:
        df = df.drop(columns=sorted(to_drop), errors="ignore")
    return df


def pick_data_file(base: Path, explicit_path: Optional[str], target_col: str) -> Path:
    """Choose a raw CSV/Parquet that actually contains the target column.
       Avoids files that look like metrics/summary reports."""
    # If user gave a path, use it
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {explicit_path}")
        return p

    # Otherwise search for a parquet/csv that likely contains the target
    candidates = [p for p in base.glob("*.parquet")] + [p for p in base.glob("*.csv")]

    # Avoid files that look like metrics/summary
    bad_hint = re.compile(r"(lime|metrics|summary|report|dice)", re.I)
    candidates = [p for p in candidates if not bad_hint.search(p.stem)]

    if not candidates:
        raise FileNotFoundError(
            "No suitable data file found. Place a raw CSV/Parquet next to the script "
            "or pass --data path/to/file."
        )

    # Prefer parquet first
    for prefer_parquet in (True, False):
        for p in candidates:
            if prefer_parquet and p.suffix.lower() != ".parquet":
                continue
            try:
                if p.suffix.lower() == ".parquet":
                    df_head = pd.read_parquet(p, engine="pyarrow")
                    cols = df_head.columns
                else:
                    df_head = pd.read_csv(p, nrows=5)
                    cols = df_head.columns
            except Exception:
                # Fallback read
                df_head = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, nrows=5)
                cols = df_head.columns

            if target_col in cols:
                return p

    # If we get here, none had the target col
    raise KeyError(
        f"No CSV/Parquet with target column '{target_col}' found. "
        "Use --data to point to your raw dataset and/or --target to set the target name."
    )


def rebuild_splits(base: Path, data_path: Optional[str], target_name: str, id_name: str
                   ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Rebuild the same split logic used previously.
       Returns (X_train, y_train, X_val, y_val)."""
    raw_path = pick_data_file(base, data_path, target_name)
    log(f"Loading data from: {raw_path.name}")

    if raw_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_csv(raw_path)

    df = standardize_cols(df)
    df = drop_anonymized_cols(df, ANON_PATTERNS)

    if target_name not in df.columns:
        raise KeyError(
            f"Target column '{target_name}' not in {raw_path.name}. "
            "Use --target to set the correct column name."
        )

    # Split columns
    y = df[target_name].astype(int)
    feature_drop = [target_name]
    if id_name in df.columns:
        feature_drop.append(id_name)
    X = df.drop(columns=feature_drop, errors="ignore")

    # Optional size cap
    if SUBSET_SIZE and len(df) > SUBSET_SIZE:
        keep_frac = SUBSET_SIZE / len(df)
        sss_keep = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - keep_frac), random_state=RANDOM_STATE
        )
        keep_idx, _ = next(sss_keep.split(X, y))
        X = X.iloc[keep_idx].reset_index(drop=True)
        y = y.iloc[keep_idx].reset_index(drop=True)
        log(f"Downsampled to {len(X)} rows for speed.")

    # Outer split: train+val vs test
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    X_trainval, _ = X.iloc[trainval_idx], X.iloc[test_idx]
    y_trainval, _ = y.iloc[trainval_idx], y.iloc[test_idx]

    # Inner split: train vs val
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION_OF_TEMP, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), \
           X_val.reset_index(drop=True), y_val.reset_index(drop=True)


def to_dense(preprocessor, X_df: pd.DataFrame) -> np.ndarray:
    Xt = preprocessor.transform(X_df)
    return Xt.toarray() if issparse(Xt) else Xt


# --------------------------------------------------------------------------------------
# Feature-name extraction 
# --------------------------------------------------------------------------------------
def get_feature_names(preprocessor, X_sample_df: pd.DataFrame) -> List[str]:

    # 1) Modern sklearn path
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        pass

    names: List[str] = []
    used_cols: List[str] = []

    # 2) Try to dig into ColumnTransformer (optionally inside a Pipeline)
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
        # Skip dropped transformers
        if trans == "drop":
            continue

        # If this transformer is a Pipeline, get its last step
        final_step = trans
        if isinstance(trans, Pipeline):
            try:
                final_step = trans.steps[-1][1]
            except Exception:
                final_step = trans

        # Resolve columns this transformer applies to
        if cols is None:

            cols = [c for c in X_sample_df.columns if c not in used_cols]
        else:
            if isinstance(cols, (list, tuple)):
                used_cols.extend(cols)

        # OneHotEncoder: create col__category names
        if isinstance(final_step, OneHotEncoder):
            if isinstance(cols, (list, tuple)):
                cats_list = final_step.categories_
                for i, col in enumerate(cols):
                    cats = cats_list[i]
                    names.extend([f"{col}__{str(cat)}" for cat in cats])
            else:
                
                feat_names = X_sample_df.columns[cols].tolist()
                cats_list = final_step.categories_
                for i, col in enumerate(feat_names):
                    cats = cats_list[i]
                    names.extend([f"{col}__{str(cat)}" for cat in cats])
        else:
            
            if isinstance(cols, (list, tuple)):
                names.extend(list(cols))
            else:
                names.extend(X_sample_df.columns[cols].tolist())

 
    try:
        Xt = preprocessor.transform(X_sample_df.head(1))
        dim = Xt.shape[1]
    except Exception:
    
        return [f"f{i}" for i in range(len(names))] if names else ["f0"]

    if len(names) != dim:

        return [f"f{i}" for i in range(dim)]

    return names


# --------------------------------------------------------------------------------------
# LIME helpers
# --------------------------------------------------------------------------------------
def kernel_width_from_scale(scale: float, d: int) -> float:
    
    return float(scale) * np.sqrt(d)


def predict_proba_wrapper(model):
    
    def _fn(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)
    return _fn


def build_lime_explainer(
    X_train_dense: np.ndarray,
    feature_names: List[str],
    kernel_width: float,
    random_state: int
) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train_dense,
        feature_names=feature_names,
        class_names=["non-fraud", "fraud"],
        mode="classification",
        discretize_continuous=False,
        sample_around_instance=True,
        kernel_width=kernel_width,
        random_state=random_state,
    )


def explain_with_compat(
    explainer: LimeTabularExplainer,
    data_row: np.ndarray,
    model,
    num_features: int,
    num_samples: int,
    labels: tuple,
    feature_selection: Optional[str] = None,
):

    sig = inspect.signature(explainer.explain_instance)
    kwargs = dict(
        data_row=data_row,
        predict_fn=predict_proba_wrapper(model),
        num_features=num_features,
        labels=labels,
        num_samples=num_samples,
    )
    if "feature_selection" in sig.parameters and feature_selection is not None:
        kwargs["feature_selection"] = feature_selection
    return explainer.explain_instance(**kwargs)


def explain_validation_subset(
    model_name: str,
    model,
    explainers: List[LimeTabularExplainer],
    X_val_dense: np.ndarray,
    feature_names: List[str],
    base: Path,
    top_k: int = TOP_K,
    num_samples_lime: int = LIME_NUM_SAMPLES,
    feature_selection: Optional[str] = "lasso_path",
    n_samples: int = N_SAMPLES,
    eval_indices: Optional[np.ndarray] = None,   # NEW
) -> None:


    n = X_val_dense.shape[0]

    # If eval_indices is provided (from SHAP metrics), use them instead of random sampling
    if eval_indices is not None:
        sel_idx = np.asarray(
            [i for i in eval_indices if 0 <= i < n],
            dtype=int,
        )
        log(f"{model_name}: using {len(sel_idx)} eval indices from SHAP.")
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        sel_idx = rng.choice(n, size=min(n_samples, n), replace=False)
        log(f"{model_name}: randomly selected {len(sel_idx)} validation rows for LIME.")


    contrib_matrix = np.zeros((len(sel_idx), X_val_dense.shape[1]), dtype=float)
    rows = []

    # Predict proba once for convenience
    pred_proba_all = model.predict_proba(X_val_dense[sel_idx])[:, 1]

    for r, (row_id, p1) in enumerate(zip(sel_idx, pred_proba_all)):
        x = X_val_dense[row_id]
        contrib_accum = np.zeros(X_val_dense.shape[1], dtype=float)

        # Bagged-LIME: average across multiple explainers (different seeds)
        for expl in explainers:
            exp = explain_with_compat(
                explainer=expl,
                data_row=x,
                model=model,
                num_features=LIME_NUM_FEATURES,
                num_samples=num_samples_lime,
                labels=(1,),
                feature_selection=feature_selection,  
            )
            asmap = exp.as_map().get(1, [])
            for j, w in asmap:
                if 0 <= j < contrib_accum.shape[0]:
                    contrib_accum[j] += w

        # Average the contributions
        if len(explainers) > 0:
            contrib_accum /= float(len(explainers))

        # Store the dense contribution row
        contrib_matrix[r] = contrib_accum

        # Export top-k by absolute weight
        nz = np.nonzero(contrib_accum)[0]
        if nz.size > 0:
            pairs_sorted = sorted(
                ((j, contrib_accum[j]) for j in nz),
                key=lambda t: abs(t[1]),
                reverse=True
            )[:top_k]
            idxs = [int(j) for j, _ in pairs_sorted]
            ws = [float(w) for _, w in pairs_sorted]
            names = [feature_names[j] if 0 <= j < len(feature_names) else f"f{j}" for j in idxs]
        else:
            idxs, ws, names = [], [], []

        rows.append({
            "row_id": int(row_id),
            "pred_proba": float(p1),
            "topk_indices": "|".join(map(str, idxs)),
            "topk_names": "|".join(names),
            "topk_weights": "|".join([f"{w:.6f}" for w in ws]),
        })

    # Save dense contributions
    npy_path = base / f"{model_name}_lime_contrib_matrix.npy"
    np.save(npy_path, contrib_matrix)
    log(f"Saved: {npy_path.name} (shape={contrib_matrix.shape})")

    # Save feature names (one per line)
    txt_path = base / f"{model_name}_feature_names.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for nm in feature_names:
            f.write(str(nm) + "\n")
    log(f"Saved: {txt_path.name}")

    # Save per-row top-k CSV
    csv_path = base / f"{model_name}_lime_local_topk.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"Saved: {csv_path.name} (rows={len(rows)})")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Path to raw CSV/Parquet with features + target.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET,
                        help="Name of target column (default: isFraud).")
    parser.add_argument("--id", type=str, default=DEFAULT_ID,
                        help="Optional ID column name (default: TransactionID).")
    args = parser.parse_args()

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
        raise FileNotFoundError("preprocessor.joblib not found in the working directory.")
    preprocessor = joblib.load(prep_path)

    # Build splits using the correct data file and target
    X_train, y_train, X_val, y_val = rebuild_splits(
        base=base,
        data_path=args.data,
        target_name=args.target,
        id_name=args.id,
    )

    # Dense matrices for LIME
    Xtr = to_dense(preprocessor, X_train)
    Xvl = to_dense(preprocessor, X_val)
    d = Xvl.shape[1]

    # Post-transform feature names
    feature_names = get_feature_names(preprocessor, X_train)

    # -------------------- XGBoost --------------------
    if xgb_path.exists():
        try:
            log("Loading XGBoost model...")
            xgb_model = joblib.load(xgb_path)

        # Sanity check: has predict_proba?
            if not hasattr(xgb_model, "predict_proba"):
                raise AttributeError(
                    "Loaded XGBoost model has no predict_proba(). "
                    "Ensure it's an XGBClassifier or compatible wrapper."
                )

            log("Running LIME for XGBoost...")
            kw_xgb = kernel_width_from_scale(KERNEL_WIDTH_SCALE["xgb"], d)
            xgb_explainers = [
                build_lime_explainer(Xtr, feature_names, kw_xgb, rs) for rs in BAG_SEEDS
            ]
                        # Try to reuse the same eval indices as SHAP metrics (if available)
            xgb_idx_path = base / "xgb_shap_eval_indices.npy"
            if xgb_idx_path.exists():
                xgb_eval_indices = np.load(xgb_idx_path)
                log(f"Loaded XGB eval indices from {xgb_idx_path.name} (n={len(xgb_eval_indices)})")
            else:
                xgb_eval_indices = None
                log("No xgb_shap_eval_indices.npy found; falling back to random sampling for LIME.")

            explain_validation_subset(
                model_name="xgb",
                model=xgb_model,
                explainers=xgb_explainers,
                X_val_dense=Xvl,
                feature_names=feature_names,
                base=base,
                top_k=TOP_K,
                num_samples_lime=LIME_NUM_SAMPLES,
                feature_selection="lasso_path",   # used only if your LIME supports it
                n_samples=N_SAMPLES,
                eval_indices=xgb_eval_indices,
            )
        except Exception as e:
            # We keep going so LR can still run, but tell you exactly why XGB failed.
            log(f"XGBoost LIME failed: {type(e).__name__}: {e}")
    else:
        log("model_xgb.joblib not found; skipping XGBoost.")


    # ---------------- Logistic Regression ----------------
    if lr_path.exists():
        lr_model = joblib.load(lr_path)
        log("Running LIME for Logistic Regression...")
        kw_lr = kernel_width_from_scale(KERNEL_WIDTH_SCALE["lr"], d)
        lr_explainers = [
            build_lime_explainer(Xtr, feature_names, kw_lr, rs) for rs in BAG_SEEDS
        ]
                # Try to reuse the same eval indices as SHAP metrics (if available)
        lr_idx_path = base / "lr_shap_eval_indices.npy"
        if lr_idx_path.exists():
            lr_eval_indices = np.load(lr_idx_path)
            log(f"Loaded LR eval indices from {lr_idx_path.name} (n={len(lr_eval_indices)})")
        else:
            lr_eval_indices = None
            log("No lr_shap_eval_indices.npy found; falling back to random sampling for LIME.")

        explain_validation_subset(
            model_name="lr",
            model=lr_model,
            explainers=lr_explainers,
            X_val_dense=Xvl,
            feature_names=feature_names,
            base=base,
            top_k=TOP_K,
            num_samples_lime=LIME_NUM_SAMPLES,
            feature_selection="lasso_path",   # used only if your LIME supports it
            n_samples=N_SAMPLES,
            eval_indices=lr_eval_indices,
        )
    else:
        log("model_logreg.joblib not found; skipping Logistic Regression.")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
