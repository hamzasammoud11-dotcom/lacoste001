# deeppurpose001.py
# Fix API mismatch: DBTA n'a pas .evaluate() -> on fait predict() + métriques

import os
import math
import numpy as np
import pandas as pd

from DeepPurpose import dataset, utils
from DeepPurpose import DTI as models


# =========================
# 1) Config projet
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DRUG_ENCODING = "Morgan"
TARGET_ENCODING = "CNN"

SPLIT_METHOD = "random"
FRAC_TRAIN, FRAC_VAL, FRAC_TEST = 0.8, 0.1, 0.1
RANDOM_SEED = 1

TRAIN_EPOCH = 10
BATCH_SIZE = 256
LR = 1e-4


# =========================
# 2) Helpers métriques
# =========================
def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(np.asarray(a, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(b, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    return pearson_corr(a, b)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def concordance_index_fast(y_true: np.ndarray, y_pred: np.ndarray, max_n: int = 2000) -> float:
    """
    CI = concordance index.
    O(n^2) si on le fait complet; on sous-échantillonne si test trop grand.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = y_true.size
    if n < 2:
        return float("nan")

    # Sous-échantillonnage pour garder un temps raisonnable sur Windows/CPU
    if n > max_n:
        rng = np.random.default_rng(0)
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
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            prod = diff_true * diff_pred
            if prod > 0:
                conc += 1.0
            elif prod == 0:
                conc += 0.5
    if total == 0:
        return float("nan")
    return float(conc / total)


# =========================
# 3) Chargement dataset
# =========================
def load_any_supported_dataset():
    """
    Essaie DAVIS puis KIBA (les deux plus utilisés dans les exemples DeepPurpose).
    """
    loaders = [
        ("DAVIS", lambda: dataset.load_process_DAVIS(path=DATA_DIR, binary=False)),
        ("KIBA",  lambda: dataset.load_process_KIBA(path=DATA_DIR, binary=False)),
    ]
    last_err = None
    for name, fn in loaders:
        try:
            X_drugs, X_targets, y = fn()
            print(f"[OK] Dataset chargé: {name} | N={len(y)}")
            return name, X_drugs, X_targets, y
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Impossible de charger DAVIS/KIBA. Dernière erreur: {last_err}")


def main():
    dataset_name, X_drugs, X_targets, y = load_any_supported_dataset()

    # =========================
    # 4) Data processing + split
    # =========================
    print("=== Data Processing ===")
    train, val, test = utils.data_process(
        X_drugs, X_targets, y,
        drug_encoding=DRUG_ENCODING,
        target_encoding=TARGET_ENCODING,
        split_method=SPLIT_METHOD,
        frac=[FRAC_TRAIN, FRAC_VAL, FRAC_TEST],
        random_seed=RANDOM_SEED
    )

    # =========================
    # 5) Modèle + training
    # =========================
    config = utils.generate_config(
        drug_encoding=DRUG_ENCODING,
        target_encoding=TARGET_ENCODING,
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=TRAIN_EPOCH,
        batch_size=BATCH_SIZE,
        LR=LR
    )

    model = models.model_initialize(**config)

    print("=== Training ===")
    # Certaines versions acceptent train(train, val, test), d'autres train(train, val)
    try:
        model.train(train, val, test)
    except TypeError:
        model.train(train, val)

    print("--- Training Finished ---")

    # =========================
    # 6) Evaluation (fix ici)
    # =========================
    print("=== Evaluation ===")
    y_true = test.Label.values
    y_pred = model.predict(test)

    # Normaliser shape
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    out_mse = mse(y_true, y_pred)
    out_rmse = float(math.sqrt(out_mse))
    out_mae = mae(y_true, y_pred)
    out_p = pearson_corr(y_true, y_pred)
    out_s = spearman_corr(y_true, y_pred)
    out_ci = concordance_index_fast(y_true, y_pred, max_n=2000)

    print(f"DATASET   = {dataset_name}")
    print(f"DRUG_ENC  = {DRUG_ENCODING}")
    print(f"TARG_ENC  = {TARGET_ENCODING}")
    print(f"TEST_MSE  = {out_mse:.6f}")
    print(f"TEST_RMSE = {out_rmse:.6f}")
    print(f"TEST_MAE  = {out_mae:.6f}")
    print(f"PEARSON   = {out_p:.6f}")
    print(f"SPEARMAN  = {out_s:.6f}")
    print(f"CI(~)     = {out_ci:.6f}  (approx si test>2000)")

    # Sauvegarde des prédictions
    pred_path = os.path.join(os.path.dirname(__file__), "predictions_test.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(pred_path, index=False)
    print(f"[OK] Prédictions sauvegardées: {pred_path}")


if __name__ == "__main__":
    main()
