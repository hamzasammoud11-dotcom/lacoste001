# run_kiba_bindingdb_tdc_deeppurpose.py
# Pipeline complet: KIBA -> BindingDB_Kd (TDC) -> DeepPurpose training/eval + exports + plots
# Outputs par dataset: predictions_test.csv, run_summary.json, scatter.png, curves_sorted.png,
# residuals.png, hist_true_pred.png, ecdf_true_pred.png

import os
import json
import math
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tdc.multi_pred import DTI
from DeepPurpose import utils
from DeepPurpose import DTI as dp_models


# -------------------------
# Metrics (régression)
# -------------------------
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def spearman(y_true, y_pred):
    a = pd.Series(np.asarray(y_true, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(y_pred, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    return pearson(a, b)

def concordance_index_approx(y_true, y_pred, max_n=2000, seed=0):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = len(y_true)
    if n < 2:
        return float("nan")
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = max_n
    conc = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1.0
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            prod = dt * dp
            if prod > 0:
                conc += 1.0
            elif prod == 0:
                conc += 0.5
    if total == 0:
        return float("nan")
    return float(conc / total)


# -------------------------
# Utils
# -------------------------
def log_section(title):
    bar = "=" * 70
    print("\n" + bar)
    print(title)
    print(bar)

def detect_cols(df: pd.DataFrame):
    cols = df.columns.tolist()
    drug_col = "Drug" if "Drug" in cols else None
    target_col = "Target" if "Target" in cols else None
    y_col = "Y" if "Y" in cols else None

    if drug_col is None:
        for c in cols:
            if "smiles" in c.lower() or "drug" in c.lower():
                drug_col = c
                break
    if target_col is None:
        for c in cols:
            if "sequence" in c.lower() or "target" in c.lower() or "protein" in c.lower():
                target_col = c
                break
    if y_col is None:
        for c in cols:
            if c.lower() in {"y", "label", "labels", "affinity"}:
                y_col = c
                break

    if drug_col is None and len(cols) >= 1:
        drug_col = cols[0]
    if target_col is None and len(cols) >= 2:
        target_col = cols[1]
    if y_col is None and len(cols) >= 3:
        y_col = cols[2]

    return drug_col, target_col, y_col

def nm_to_paffinity(y_nm):
    """
    suppose y en nM => p = 9 - log10(nM)
    """
    y_nm = np.asarray(y_nm, dtype=float)
    y_nm = np.where(y_nm <= 0, np.nan, y_nm)
    return 9.0 - np.log10(y_nm)

def make_run_dir(base_dir, dataset_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{dataset_name}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def save_plots(run_dir, y_true, y_pred):
    """
    Génère:
      - scatter.png
      - curves_sorted.png
      - residuals.png
      - hist_true_pred.png
      - ecdf_true_pred.png
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    # 1) Scatter: y_true vs y_pred
    scatter_png = os.path.join(run_dir, "scatter.png")
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Test: y_true vs y_pred")
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=200)
    plt.close()

    # 2) Courbes triées: y_true et y_pred (tri par y_true)
    curves_png = os.path.join(run_dir, "curves_sorted.png")
    order = np.argsort(y_true)
    plt.figure()
    plt.plot(y_true[order], label="y_true")
    plt.plot(y_pred[order], label="y_pred")
    plt.xlabel("samples (sorted by y_true)")
    plt.ylabel("value")
    plt.title("Test: curves (sorted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_png, dpi=200)
    plt.close()

    # 3) Résidus: (y_pred - y_true) vs y_true
    res_png = os.path.join(run_dir, "residuals.png")
    res = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, res, s=8)
    plt.axhline(0)
    plt.xlabel("y_true")
    plt.ylabel("y_pred - y_true")
    plt.title("Test: residuals")
    plt.tight_layout()
    plt.savefig(res_png, dpi=200)
    plt.close()

    # 4) Histogrammes: distribution y_true vs y_pred
    hist_png = os.path.join(run_dir, "hist_true_pred.png")
    plt.figure()
    plt.hist(y_true, bins=40, alpha=0.6, label="y_true")
    plt.hist(y_pred, bins=40, alpha=0.6, label="y_pred")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title("Test: distribution (hist)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    # 5) ECDF: courbes de distribution cumulée
    ecdf_png = os.path.join(run_dir, "ecdf_true_pred.png")

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    xt, yt = ecdf(y_true)
    xp, yp = ecdf(y_pred)

    plt.figure()
    plt.plot(xt, yt, label="y_true")
    plt.plot(xp, yp, label="y_pred")
    plt.xlabel("value")
    plt.ylabel("ECDF")
    plt.title("Test: ECDF (true vs pred)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ecdf_png, dpi=200)
    plt.close()

    print("[PLOT] saved:", scatter_png)
    print("[PLOT] saved:", curves_png)
    print("[PLOT] saved:", res_png)
    print("[PLOT] saved:", hist_png)
    print("[PLOT] saved:", ecdf_png)


def run_one_dataset(dataset_name: str, args):
    run_id, run_dir = make_run_dir(args.runs_dir, dataset_name)
    pred_path = os.path.join(run_dir, "predictions_test.csv")
    summary_path = os.path.join(run_dir, "run_summary.json")

    # -------- 1) load via TDC
    log_section(f"[{dataset_name}] 1) LOAD (TDC)")
    t0 = time.time()

    data = DTI(name=dataset_name)

    # harmonize only for BindingDB_*
    if dataset_name.startswith("BindingDB_") and args.harmonize_bindingdb == "mean":
        print("[TDC] harmonize_affinities(mode='mean')")
        data.harmonize_affinities(mode="mean")

    try:
        df = data.get_data()
    except TypeError:
        df = data.get_data(format="df")

    print(f"[INPUT] raw_shape={df.shape}")
    drug_col, target_col, y_col = detect_cols(df)
    print(f"[INPUT] detected_cols: drug='{drug_col}', target='{target_col}', label='{y_col}'")

    df = df[[drug_col, target_col, y_col]].copy()
    df.columns = ["Drug", "Target", "Y"]

    # -------- 2) clean + label transform
    log_section(f"[{dataset_name}] 2) CLEAN + LABEL TRANSFORM")
    n0 = len(df)
    df = df.dropna()
    df["Drug"] = df["Drug"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()
    df = df[(df["Drug"] != "") & (df["Target"] != "")]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna()
    n1 = len(df)

    # subsample only BindingDB_Kd (CPU friendly)
    if dataset_name == "BindingDB_Kd" and args.max_rows_bindingdb is not None:
        if len(df) > args.max_rows_bindingdb:
            df = df.sample(n=args.max_rows_bindingdb, random_state=args.seed).reset_index(drop=True)
            print(f"[SUBSAMPLE] BindingDB_Kd -> {len(df)} rows")

    if args.label_transform == "paffinity_nm":
        y_nm = df["Y"].astype(float).values
        df["Y"] = nm_to_paffinity(y_nm)

    df = df.dropna()
    n2 = len(df)

    df = df.drop_duplicates(subset=["Drug", "Target"], keep="first").reset_index(drop=True)
    n3 = len(df)

    y = df["Y"].astype(float).values
    print(f"[CLEAN] start={n0} -> after_parse={n1} -> after_transform={n2} -> after_dedup={n3}")
    print(f"[LABEL] mode={args.label_transform}")
    print(f"[LABEL] stats: min={np.min(y):.6f} | mean={np.mean(y):.6f} | median={np.median(y):.6f} | max={np.max(y):.6f}")
    print("[SAMPLE] first_rows:")
    print(df.head(3).to_string(index=False))

    # -------- 3) DeepPurpose data_process
    log_section(f"[{dataset_name}] 3) DATA_PROCESS (DeepPurpose)")
    X_drugs = df["Drug"].values
    X_targets = df["Target"].values
    y = df["Y"].values

    train, val, test = utils.data_process(
        X_drugs, X_targets, y,
        drug_encoding=args.drug_enc,
        target_encoding=args.target_enc,
        split_method="random",
        frac=[0.8, 0.1, 0.1],
        random_seed=args.seed
    )

    try:
        n_train, n_val, n_test = len(train.Label), len(val.Label), len(test.Label)
    except Exception:
        n_train, n_val, n_test = len(train), len(val), len(test)

    print(f"[SPLIT] train={n_train} | val={n_val} | test={n_test}")
    print(f"[ENC] drug_encoding={args.drug_enc} | target_encoding={args.target_enc}")

    # -------- 4) train
    log_section(f"[{dataset_name}] 4) TRAIN (DeepPurpose)")
    config = utils.generate_config(
        drug_encoding=args.drug_enc,
        target_encoding=args.target_enc,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=args.epochs,
        batch_size=args.batch,
        LR=args.lr,
        result_folder=run_dir
    )

    print("[MODEL] config:")
    print(f"  epochs={args.epochs} | batch={args.batch} | lr={args.lr}")
    print(f"  hidden=[1024,1024,512] | result_dir={run_dir}")

    model = dp_models.model_initialize(**config)

    t_train0 = time.time()
    try:
        model.train(train, val, test)
    except TypeError:
        model.train(train, val)
    t_train1 = time.time()
    print(f"[TIME] train_seconds={t_train1 - t_train0:.1f}")
    model.train(train, val, test)
    model.save_model(run_dir)
    print("[FILE] saved model in:", run_dir)


    # -------- 5) eval + export
    log_section(f"[{dataset_name}] 5) EVAL + EXPORT")
    print("[PREDICT] predicting on test...")
    y_true = np.asarray(test.Label.values, dtype=float).reshape(-1)
    y_pred = np.asarray(model.predict(test), dtype=float).reshape(-1)

    m_mse = mse(y_true, y_pred)
    m_rmse = float(math.sqrt(m_mse))
    m_mae = mae(y_true, y_pred)
    m_p = pearson(y_true, y_pred)
    m_s = spearman(y_true, y_pred)
    m_ci = concordance_index_approx(y_true, y_pred, max_n=2000, seed=args.seed)

    print("[METRICS] test:")
    print(f"  MSE        = {m_mse:.6f}")
    print(f"  RMSE       = {m_rmse:.6f}")
    print(f"  MAE        = {m_mae:.6f}")
    print(f"  Pearson    = {m_p:.6f}")
    print(f"  Spearman   = {m_s:.6f}")
    print(f"  CI(approx) = {m_ci:.6f}")

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(pred_path, index=False)
    print(f"[FILE] saved predictions: {pred_path}")

    save_plots(run_dir, y_true, y_pred)

    summary = {
        "run_id": run_id,
        "dataset": dataset_name,
        "drug_encoding": args.drug_enc,
        "target_encoding": args.target_enc,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "label_transform": args.label_transform,
        "harmonize_bindingdb": args.harmonize_bindingdb,
        "max_rows_bindingdb": args.max_rows_bindingdb,
        "n_rows_after_clean": int(len(df)),
        "metrics_test": {
            "mse": m_mse,
            "rmse": m_rmse,
            "mae": m_mae,
            "pearson": m_p,
            "spearman": m_s,
            "ci_approx": m_ci,
        },
        "files": {
            "predictions_test_csv": pred_path,
            "scatter_png": os.path.join(run_dir, "scatter.png"),
            "curves_sorted_png": os.path.join(run_dir, "curves_sorted.png"),
            "residuals_png": os.path.join(run_dir, "residuals.png"),
            "hist_true_pred_png": os.path.join(run_dir, "hist_true_pred.png"),
            "ecdf_true_pred_png": os.path.join(run_dir, "ecdf_true_pred.png"),
        },
        "timing": {
            "total_seconds": time.time() - t0,
            "train_seconds": t_train1 - t_train0,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[FILE] saved summary: {summary_path}")
    print("[DONE]", dataset_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug_enc", default="Morgan")
    ap.add_argument("--target_enc", default="CNN")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)

    # demandé: label_transform=paffinity_nm
    ap.add_argument("--label_transform", default="paffinity_nm", choices=["paffinity_nm", "none"])

    # demandé: harmonize=mean pour BindingDB
    ap.add_argument("--harmonize_bindingdb", default="mean", choices=["mean", "none"])

    # CPU-friendly (BindingDB_Kd énorme). Mets 50000 pour tester vite.
    ap.add_argument("--max_rows_bindingdb", type=int, default=200000,
                    help="None=full (risqué). Recommandé CPU: 50k-200k.")
    ap.add_argument("--runs_dir", default=os.path.join(os.path.dirname(__file__), "runs"))

    args = ap.parse_args()
    os.makedirs(args.runs_dir, exist_ok=True)

    # Pipeline demandé: KIBA puis BindingDB_Kd
    for ds in ["KIBA", "BindingDB_Kd"]:
        run_one_dataset(ds, args)


if __name__ == "__main__":
    main()
