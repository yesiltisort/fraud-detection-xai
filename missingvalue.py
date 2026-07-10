# missingness_audit.py
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Aynı pattern'ler (baseline ile uyumlu)
ANON_PATTERNS = [r"^V\d+$", r"^C\d+$", r"^D\d+$", r"^M\d+$", r"^id_\d+$"]

def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df

def drop_anonymized_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns if any(re.match(p, c) for p in ANON_PATTERNS)]
    return df.drop(columns=to_drop, errors="ignore")

def missingness_table(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = (miss_cnt / len(df) * 100).round(2)
    out = pd.DataFrame({"missing_count": miss_cnt, "missing_pct": miss_pct})
    out = out.sort_values("missing_pct", ascending=False)
    return out.head(top_n)

def sparsity_table(X: pd.DataFrame) -> dict:
    # preprocess sonrası matriste NaN beklemiyoruz; sparsity ~ 0 değer oranı
    arr = X.values
    zero_frac = float((arr == 0).mean())
    nonzero_frac = 1.0 - zero_frac
    return {"rows": X.shape[0], "cols": X.shape[1], "zero_frac": zero_frac, "nonzero_frac": nonzero_frac}

def main():
    base = Path(".")  # aynı klasörde çalıştır

    # ---------- (A) HAM VERİ: merge sonrası missingness ----------
    tx_path = base / "train_transaction.csv"
    id_path = base / "train_identity.csv"
    if tx_path.exists() and id_path.exists():
        tx = pd.read_csv(tx_path)
        idf = pd.read_csv(id_path)
        df = standardize_cols(tx.merge(idf, on="TransactionID", how="left"))
        df = drop_anonymized_cols(df)

        # hedef/id kolonlarını ayır (istersen)
        X_raw = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

        print("\n[RAW] Top missing features (merged, anonymized dropped):")
        print(missingness_table(X_raw, top_n=30).to_string())

        # “identity vs transaction” ayrımı için kabaca: identity dosyasında gelen kolonlar
        id_cols = set(standardize_cols(idf).columns) - {"TransactionID"}
        tx_cols = set(standardize_cols(tx).columns) - {"TransactionID", "isFraud"}

        id_in_merged = [c for c in X_raw.columns if c in id_cols]
        tx_in_merged = [c for c in X_raw.columns if c in tx_cols]

        id_miss = X_raw[id_in_merged].isna().mean().mean() * 100 if id_in_merged else np.nan
        tx_miss = X_raw[tx_in_merged].isna().mean().mean() * 100 if tx_in_merged else np.nan

        print(f"\n[RAW] Avg missingness (%): identity-derived cols = {id_miss:.2f} | transaction cols = {tx_miss:.2f}")
    else:
        print("\n[RAW] train_transaction.csv / train_identity.csv bulunamadı (ham missingness tablosu üretilemedi).")

    # ---------- (B) PREPROCESS SONRASI: sparsity ----------
    xtr_path = base / "X_train.csv"
    xte_path = base / "X_test.csv"
    if xtr_path.exists():
        Xtr = pd.read_csv(xtr_path)
        print("\n[PROCESSED] Sparsity summary for X_train.csv:")
        print(sparsity_table(Xtr))
    else:
        print("\n[PROCESSED] X_train.csv bulunamadı (sparsity özeti üretilemedi).")

    if xte_path.exists():
        Xte = pd.read_csv(xte_path)
        print("\n[PROCESSED] Sparsity summary for X_test.csv:")
        print(sparsity_table(Xte))

if __name__ == "__main__":
    main()
