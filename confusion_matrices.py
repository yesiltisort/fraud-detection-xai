
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).parent  


def log(msg: str) -> None:
    print(f"[info] {msg}")


def load_pos_neg_counts(base: Path):
    y_test = pd.read_csv(base / "y_test.csv")["isFraud"].astype(int)
    P = int((y_test == 1).sum())
    N = int((y_test == 0).sum())
    log(f"Test set: positives={P}, negatives={N}")
    return P, N


def confusion_from_metrics(precision: float, recall: float, P: int, N: int):

    # TP = recall * P
    TP = int(round(recall * P))
    FN = P - TP

    # precision = TP / (TP + FP) -> FP = TP * (1/precision - 1)
    FP = int(round(TP * (1.0 / precision - 1.0)))
    TN = N - FP

    return np.array([[TN, FP], [FN, TP]], dtype=int)


def plot_confusion(cm: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    light_blues = LinearSegmentedColormap.from_list(
    "light_blues",
    ["#eaf3fc", "#d6e6f8", "#bfd9f3", "#a6ccec", "#90bde4"]
)
    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap=light_blues,
        vmin=0,
        vmax=cm.max() * 0.4,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    classes = [0, 1]
    ax.set_xticks(range(2))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(2))
    ax.set_yticklabels(classes)

    cmf = cm.astype(float)
    row_sums = cmf.sum(axis=1, keepdims=True)
    norm = np.divide(
        cmf,
        np.maximum(row_sums, 1),
        out=np.zeros_like(cmf),
        where=row_sums != 0,
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(cmf[i, j])}\n({norm[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    log(f"Saved: {save_path.resolve()}")
    plt.close(fig)


def main():
    base = BASE
    log(f"Working directory: {base.resolve()}")

    P, N = load_pos_neg_counts(base)


    lr_prec = 0.196970
    lr_rec = 0.260000

    xgb_prec = 0.433775
    xgb_rec = 0.374286

    cm_lr = confusion_from_metrics(lr_prec, lr_rec, P, N)
    cm_xgb = confusion_from_metrics(xgb_prec, xgb_rec, P, N)

    log(f"LR confusion (from table):\n{cm_lr}")
    log(f"XGB confusion (from table):\n{cm_xgb}")

    # 4) Grafikler
    plot_confusion(
        cm_lr,
        "Logistic Regression — Confusion Matrix (Test, from table metrics)",
        base / "cm_logreg_from_table.png",
    )

    plot_confusion(
        cm_xgb,
        "XGBoost — Confusion Matrix (Test, from table metrics)",
        base / "cm_xgb_from_table.png",
    )


if __name__ == "__main__":
    main()
