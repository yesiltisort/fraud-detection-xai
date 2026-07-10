import joblib
import numpy as np
import pandas as pd
import dice_ml

from dice_ml import Dice
from pathlib import Path
from typing import Optional

# 1. Import preprocessing-based loader (train/test split in model space)
from preprocess_baseline2 import load_preprocessed_data


def load_feature_names(n_features: int) -> list:


    base = Path.cwd()
    candidates = ["lr_feature_names.txt", "xgb_feature_names.txt"]
    for fname in candidates:
        p = base / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            if len(names) == n_features:
                print(f"[info] Loaded feature names from {fname}")
                return names
            else:
                print(
                    f"[warn] {fname} has {len(names)} names, but data has {n_features} features. Ignoring."
                )

    # Fallback: generic names
    print("[warn] Could not load feature names, falling back to f0..fN.")
    return [f"f{i}" for i in range(n_features)]


def build_dice_data_interface():

    X_train, X_test, y_train, y_test = load_preprocessed_data()

 
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    if not isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = pd.Series(y_train, name="isFraud")
    if not isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = pd.Series(y_test, name=y_train.name)


    n_features = X_train.shape[1]
    feature_names = load_feature_names(n_features)

    X_train.columns = feature_names
    X_test.columns = feature_names

    target_col = y_train.name if y_train.name is not None else "isFraud"

 
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    continuous_features = feature_names  

    data_dice = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name=target_col,
    )

    return data_dice, X_test, y_test, target_col, feature_names


def load_models(lr_path="model_logreg.joblib", xgb_path="model_xgb.joblib"):


    lr_model = joblib.load(lr_path)
    xgb_model = joblib.load(xgb_path)
    return lr_model, xgb_model


def select_query_instances_from_shap_indices(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_idx_file: str,
    sample_size: int,
) -> pd.DataFrame:

    base = Path.cwd()
    idx_path = base / shap_idx_file


    X_test = X_test.copy()
    X_test["row_id"] = np.arange(len(X_test))

    if not isinstance(y_test, (pd.Series, pd.DataFrame)):
        y_test = pd.Series(y_test, name="isFraud")


    if not idx_path.exists():
        print(f"[warn] {model_name}: {shap_idx_file} not found. Using first {sample_size} rows.")
        query_df = X_test.iloc[:sample_size].copy()
        query_df[y_test.name] = y_test.iloc[:sample_size].values
        return query_df

    shap_indices = np.load(idx_path)
    shap_indices = np.asarray(shap_indices, dtype=int)


    valid_indices = [i for i in shap_indices if 0 <= i < len(X_test)]
    if len(valid_indices) == 0:
        print(
            f"[warn] {model_name}: No valid indices from {shap_idx_file} "
            f"within X_test range. Using first {sample_size} rows."
        )
        query_df = X_test.iloc[:sample_size].copy()
        query_df[y_test.name] = y_test.iloc[:sample_size].values
        return query_df


    subset = X_test.loc[valid_indices].copy()
    subset[y_test.name] = y_test.loc[valid_indices].values


    print(
        f"[info] {model_name}: loaded {len(valid_indices)} SHAP indices from {shap_idx_file}; "
        f"will later keep at most {sample_size} for DiCE."
    )
    return subset


def generate_and_save_dice_for_model(
    model_name: str,
    model,
    data_dice,
    base_query_df: pd.DataFrame,
    feature_names,
    output_prefix: str,
    sample_size: int = 100,
    total_cfs: int = 1,
    method: str = "random"
):
    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",
    )

    dice = Dice(
        data_dice,
        dice_model,
        method="random", 
    )

    df = base_query_df.copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    target_col = [c for c in df.columns if c not in feature_names and c != "row_id"]
    if len(target_col) == 0:
        raise ValueError(f"[{model_name}] Could not infer target column in base_query_df.")
    target_col = target_col[0]


    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(df[feature_names])[:, 1]
    else:
        y_pred_proba = model.predict(df[feature_names])

    df[f"{model_name}_pred_proba"] = y_pred_proba


    if len(df) > sample_size:
        df = df.sort_values(by=f"{model_name}_pred_proba", ascending=False).head(sample_size)

    df = df.reset_index(drop=True)

    print(
        f"[info] {model_name}: using {len(df)} query instances for DiCE "
        f"(max={sample_size})."
    )


    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",  
    )

    dice = Dice(
        data_dice,
        dice_model,
        method="random", 
    )


    query_instances = df[feature_names].copy()


    dice_exp = dice.generate_counterfactuals(
        query_instances,
        total_CFs=total_cfs,
        desired_class="opposite",
    )

    cf_rows = []
    for i, cf_example in enumerate(dice_exp.cf_examples_list):
        cf_df = cf_example.final_cfs_df.copy()


        source_row_id = int(df["row_id"].iloc[i])
        cf_df["row_id"] = source_row_id
        cf_df["query_index"] = i


        if hasattr(model, "predict_proba"):
            cf_pred_proba = model.predict_proba(cf_df[feature_names])[:, 1]
        else:
            cf_pred_proba = model.predict(cf_df[feature_names])
        cf_df[f"{model_name}_cf_pred_proba"] = cf_pred_proba

        cf_rows.append(cf_df)

    if len(cf_rows) == 0:
        print(f"[{model_name}] No counterfactuals were generated.")
        return

    cf_all = pd.concat(cf_rows, ignore_index=True)


    query_out = df.copy()

    query_filename = f"{output_prefix}_{model_name}_dice_queries.csv"
    cf_filename = f"{output_prefix}_{model_name}_dice_cfs.csv"

    query_out.to_csv(query_filename, index=False)
    cf_all.to_csv(cf_filename, index=False)

    print(f"[{model_name}] Saved query instances to: {query_filename}")
    print(f"[{model_name}] Saved counterfactuals to: {cf_filename}")


if __name__ == "__main__":
    base = Path.cwd()
    print(f"[info] Working directory: {base.resolve()}")

    data_dice, X_test, y_test, target_col, feature_names = build_dice_data_interface()

    lr_model, xgb_model = load_models(
        lr_path="model_logreg.joblib",
        xgb_path="model_xgb.joblib",
    )

    # LR
    lr_base_queries = select_query_instances_from_shap_indices(
        model_name="LR",
        X_test=X_test,
        y_test=y_test,
        shap_idx_file="lr_shap_eval_indices.npy",
        sample_size=100,  
    )

    # XGB
    xgb_base_queries = select_query_instances_from_shap_indices(
        model_name="XGB",
        X_test=X_test,
        y_test=y_test,
        shap_idx_file="xgb_shap_eval_indices.npy",
        sample_size=100,  
    )
  
  
    if "isFraud" in xgb_base_queries.columns:
        xgb_base_queries = xgb_base_queries[xgb_base_queries["isFraud"] == 1].copy()


    xgb_base_queries = xgb_base_queries.reset_index(drop=True)
    print("[debug] XGB base_queries shape:", xgb_base_queries.shape)
    print("[debug] XGB class distribution:", xgb_base_queries["isFraud"].value_counts())

  
    generate_and_save_dice_for_model(
        model_name="LR",
        model=lr_model,
        data_dice=data_dice,
        base_query_df=lr_base_queries,
        feature_names=feature_names,
        output_prefix="dice_results",
        sample_size=100,
        total_cfs=1,
        method="random",
    )

    generate_and_save_dice_for_model(
        model_name="XGB",
        model=xgb_model,
        data_dice=data_dice,
        base_query_df=xgb_base_queries,
        feature_names=feature_names,
        output_prefix="dice_results",
        sample_size=10,
        total_cfs=1,
        method="kdtree",
    )

    print("Done generating DiCE explanations for LR and XGB.")
