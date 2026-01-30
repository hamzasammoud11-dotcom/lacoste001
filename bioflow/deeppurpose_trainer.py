import os
import json
import math
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import pickle


import numpy as np
import pandas as pd

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
    """
    CI approximatif (échantillonné si trop grand).
    """
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

def detect_cols(df):
    cols = df.columns.tolist()
    # TDC DTI: souvent Drug, Target, Y
    drug_col = "Drug" if "Drug" in cols else None
    target_col = "Target" if "Target" in cols else None
    y_col = "Y" if "Y" in cols else None

    # fallback "best effort"
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

    # dernier fallback: 0,1,2
    if drug_col is None and len(cols) >= 1:
        drug_col = cols[0]
    if target_col is None and len(cols) >= 2:
        target_col = cols[1]
    if y_col is None and len(cols) >= 3:
        y_col = cols[2]

    return drug_col, target_col, y_col

def label_transform(y, mode, dataset_name):
    y = np.asarray(y, dtype=float)

    # Force auto to paffinity_nm for standard datasets
    if mode == "auto":
        if dataset_name.lower() in ["davis", "kiba"] or dataset_name.startswith("BindingDB"):
            mode = "paffinity_nm"
        else:
            mode = "none"

    if mode == "none":
        return y, "none"

    if mode == "paffinity_nm":
        # SAFETY CHECK: Clip values to avoid log(0) or log(negative)
        y = np.where(y < 1e-9, 1e-9, y) 
        # Convert nM to pM ( -log10( Molar ) )
        # Value 100 nM = 100e-9 M = 1e-7 M -> -log10(1e-7) = 7.0
        # Formula: 9 - log10(nM)
        y = 9.0 - np.log10(y)
        return y, "paffinity_nm"

    raise ValueError(f"Unknown label_transform: {mode}")

def make_run_dir(base_dir, dataset):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{dataset}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def check_gpu():
    """Check GPU availability and return CUDA ID for DeepPurpose."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[SYSTEM] ✅ CUDA GPU Detected: {device_name}")
        print(f"[SYSTEM]    Memory: {gpu_memory:.1f} GB")
        print(f"[SYSTEM]    CUDA Version: {torch.version.cuda}")
        return 0  # Use GPU 0
    else:
        print("\n[SYSTEM] ⚠️ No GPU detected. Running on CPU (will be slow).")
        return -1  # Use CPU


def main():
    # --- GPU DETECTION FIRST ---
    use_cuda = check_gpu()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DAVIS", help="DAVIS | KIBA | BindingDB_Kd | BindingDB_Ki | BindingDB_IC50")
    ap.add_argument("--drug_enc", default="Morgan")
    ap.add_argument("--target_enc", default="CNN")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="random", help="DeepPurpose split_method (random recommandé ici)")
    ap.add_argument("--frac_train", type=float, default=0.8)
    ap.add_argument("--frac_val", type=float, default=0.1)
    ap.add_argument("--frac_test", type=float, default=0.1)
    ap.add_argument("--label_transform", default="paffinity_nm", help="Force log transform! (auto | none | paffinity_nm)")
    ap.add_argument("--harmonize", default="none", help="none | mean | max_affinity (utile surtout BindingDB_*)")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--dry_run", action="store_true", help="Charge dataset + prints info, sans training")
    ap.add_argument("--gpu", type=int, default=None, help="Override GPU ID (default: auto-detect)")
    args = ap.parse_args()

    # Override GPU if specified
    if args.gpu is not None:
        use_cuda = args.gpu
        print(f"[SYSTEM] Using specified GPU: {use_cuda}")

    np.random.seed(args.seed)

    run_id, run_dir = make_run_dir(args.runs_dir, args.dataset)
    pred_path = os.path.join(run_dir, "predictions_test.csv")
    summary_path = os.path.join(run_dir, "run_summary.json")

    # -------------------------
    # 1) LOAD DATA (TDC)
    # -------------------------
    log_section("[1] LOAD DATASET (TDC)")
    print(f"[RUN] run_id={run_id}")
    print(f"[RUN] dataset={args.dataset}")
    print(f"[RUN] cache=PyTDC (download auto si nécessaire)")

    t0 = time.time()
    data = DTI(name=args.dataset)

    if args.harmonize != "none" and args.dataset.startswith("BindingDB_"):
        mode = args.harmonize
        print(f"[TDC] harmonize_affinities(mode='{mode}')")
        data.harmonize_affinities(mode=mode)

    # PyTDC renvoie un DataFrame via get_data() (selon version, get_data(format='df') existe aussi)
    try:
        df = data.get_data()
    except TypeError:
        df = data.get_data(format="df")

    print(f"[INPUT] raw_shape={df.shape}")
    drug_col, target_col, y_col = detect_cols(df)
    print(f"[INPUT] detected_cols: drug='{drug_col}', target='{target_col}', label='{y_col}'")

    # -------------------------
    # 2) CLEAN + PREP
    # -------------------------
    log_section("[2] CLEAN + PREP (Input DeepPurpose)")
    df = df[[drug_col, target_col, y_col]].copy()
    df.columns = ["Drug", "Target", "Y"]

    n0 = len(df)
    df = df.dropna()
    df["Drug"] = df["Drug"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()
    df = df[(df["Drug"] != "") & (df["Target"] != "")]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna()
    n1 = len(df)

    y_raw = df["Y"].astype(float).values
    y_trans, used_transform = label_transform(y_raw, args.label_transform, args.dataset)
    df["Y"] = y_trans
    df = df.dropna()
    n2 = len(df)

    # drop duplicates (Drug,Target) -> keep first
    df = df.drop_duplicates(subset=["Drug", "Target"], keep="first").reset_index(drop=True)
    n3 = len(df)

    print(f"[CLEAN] start={n0} -> after_na/parse={n1} -> after_transform={n2} -> after_dedup={n3}")
    print(f"[LABEL] transform={used_transform}")
    y = df["Y"].astype(float).values
    print(f"[LABEL] stats: min={np.min(y):.6f} | mean={np.mean(y):.6f} | median={np.median(y):.6f} | max={np.max(y):.6f}")
    print("[SAMPLE] first_rows:")
    print(df.head(3).to_string(index=False))

    if args.dry_run:
        print(f"\n[DRY_RUN] stop ici. (Aucun training lancé)")
        return

    # -------------------------
    # 3) SPLIT + ENCODE (DeepPurpose)
    # -------------------------
    log_section("[3] DATA_PROCESS (DeepPurpose)")
    frac = [args.frac_train, args.frac_val, args.frac_test]
    print(f"[SPLIT] method={args.split} | frac={frac} | seed={args.seed}")
    X_drugs = df["Drug"].values
    X_targets = df["Target"].values
    y = df["Y"].values

    train, val, test = utils.data_process(
        X_drugs, X_targets, y,
        drug_encoding=args.drug_enc,
        target_encoding=args.target_enc,
        split_method=args.split,
        frac=frac,
        random_seed=args.seed
    )

    # tailles (DeepPurpose objects ont souvent .Label)
    try:
        n_train, n_val, n_test = len(train.Label), len(val.Label), len(test.Label)
    except Exception:
        # fallback: len(obj)
        n_train, n_val, n_test = len(train), len(val), len(test)

    print(f"[SPLIT] sizes: train={n_train} | val={n_val} | test={n_test}")
    print(f"[ENC] drug_encoding={args.drug_enc} | target_encoding={args.target_enc}")

    # -------------------------
    # 4) MODEL INIT + TRAIN (WITH GPU CONFIG)
    # -------------------------
    log_section("[4] MODEL INIT + TRAIN")
    config = utils.generate_config(
        drug_encoding=args.drug_enc,
        target_encoding=args.target_enc,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=args.epochs,
        batch_size=args.batch,
        LR=args.lr,
        result_folder=run_dir,
        cuda_id=use_cuda  # <<< GPU CONFIG HERE
    )
    # --- Persist config for API (server/api.py expects config.pkl) ---
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"[FILE] saved config: {os.path.join(run_dir, 'config.pkl')}")


    print("[MODEL] config:")
    print(f"  epochs={args.epochs} | batch={args.batch} | lr={args.lr}")
    print(f"  hidden=[1024,1024,512] | result_dir={run_dir}")
    print(f"  cuda_id={use_cuda} {'(GPU)' if use_cuda >= 0 else '(CPU)'}")

    model = dp_models.model_initialize(**config)
    
    # Verify device placement
    if use_cuda >= 0:
        device = next(model.model.parameters()).device
        print(f"[MODEL] ✓ Model loaded on device: {device}")

    t_train0 = time.time()
    try:
        model.train(train, val, test)
    except TypeError:
        model.train(train, val)
    t_train1 = time.time()
    print(f"[TIME] train_seconds={t_train1 - t_train0:.1f}")
    # --- Persist weights for API (server/api.py expects model.pt) ---
    model_path = os.path.join(run_dir, "model.pt")

    try:
        model.save_model(run_dir)  # DeepPurpose-style save (often writes model.pt in the folder)
    except Exception as e:
        print(f"[WARNING] model.save_model(run_dir) failed: {e}")

    # Fallback if model.pt not created
    if not os.path.exists(model_path):
        try:
            model.save_model(model_path)
        except Exception as e:
            print(f"[WARNING] model.save_model(model_path) failed: {e}")

    print(f"[FILE] model.pt exists? {os.path.exists(model_path)} -> {model_path}")


    # -------------------------
    # 5) EVAL + EXPORT
    # -------------------------
    log_section("[5] EVAL + EXPORT")
    print("[PREDICT] predicting on test...")
    
    # [FIX] Reset index to ensure alignment between DataFrame and Model Output
    test = test.reset_index(drop=True)
    
    y_true = np.asarray(test.Label.values, dtype=float).reshape(-1)
    
    # [DEBUG] Check what the model is actually predicting
    raw_pred = model.predict(test)
    y_pred = np.asarray(raw_pred, dtype=float).reshape(-1)

    # [DEBUG] Print first 5 comparisons to verify scaling matches
    print(f"[DEBUG] First 5 True: {y_true[:5]}")
    print(f"[DEBUG] First 5 Pred: {y_pred[:5]}")

    m_mse = mse(y_true, y_pred)
    m_rmse = float(math.sqrt(m_mse))
    m_mae = mae(y_true, y_pred)
    m_p = pearson(y_true, y_pred)
    m_s = spearman(y_true, y_pred)
    m_ci = concordance_index_approx(y_true, y_pred, max_n=2000, seed=args.seed)

    print("[METRICS] test:")
    print(f"  MSE   = {m_mse:.6f}")
    print(f"  RMSE  = {m_rmse:.6f}")
    print(f"  MAE   = {m_mae:.6f}")
    print(f"  Pearson  = {m_p:.6f}")
    print(f"  Spearman = {m_s:.6f}")
    print(f"  CI(approx) = {m_ci:.6f}")

    out_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    out_pred.to_csv(pred_path, index=False)
    print(f"[FILE] saved predictions: {pred_path}")

    summary = {
        "run_id": run_id,
        "dataset": args.dataset,
        "drug_encoding": args.drug_enc,
        "target_encoding": args.target_enc,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "split_method": args.split,
        "frac": [args.frac_train, args.frac_val, args.frac_test],
        "label_transform": used_transform,
        "harmonize": args.harmonize,
        "n_rows_after_clean": int(len(df)),
        "cuda_id": use_cuda,
        "gpu_used": use_cuda >= 0,
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda >= 0 else "CPU",
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
        },
        "timing": {
            "load_seconds": time.time() - t0,
            "train_seconds": t_train1 - t_train0,
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[FILE] saved summary: {summary_path}")

    # -------------------------
    # 6) VISUALISATION
    # -------------------------
    log_section("[6] VISUALISATION")
    
    # Scatter plot
    scatter_png = os.path.join(run_dir, "scatter.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect fit')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Test: y_true vs y_pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=200)
    plt.close()
    print(f"[PLOT] saved: {scatter_png}")

    # Sorted curves
    curves_png = os.path.join(run_dir, "curves_sorted.png")
    order = np.argsort(y_true)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[order], label="y_true", alpha=0.7)
    plt.plot(y_pred[order], label="y_pred", alpha=0.7)
    plt.xlabel("samples (sorted by y_true)")
    plt.ylabel("value")
    plt.title("Test: curves (sorted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_png, dpi=200)
    plt.close()
    print(f"[PLOT] saved: {curves_png}")

    # Residuals
    res_png = os.path.join(run_dir, "residuals.png")
    res = y_pred - y_true
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, res, s=8, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("y_true")
    plt.ylabel("y_pred - y_true")
    plt.title("Test: residuals")
    plt.tight_layout()
    plt.savefig(res_png, dpi=200)
    plt.close()
    print(f"[PLOT] saved: {res_png}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()