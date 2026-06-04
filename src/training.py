"""
training.py
-----------
Cross-validation training loop, early-stopping helpers, metrics logging,
and loss-curve plotting for the CGTST pipeline.
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import CGTST, arrange_input, regularize, ridge_regularize


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss(history: dict, model_type: str = "cgtst", folder: str = "plots") -> None:
    """
    Save a training/validation loss curve as PNG and append it to a JSON log.

    Args:
        history (dict): Must contain at least one of 'train_loss' / 'val_loss'.
        model_type (str): Identifier used in the filename.
        folder (str): Output directory.
    """
    plt.figure(figsize=(10, 6))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Train and Validation Loss per Epoch — {model_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if history.get("train_loss"):
        plt.ylim([0, max(history["train_loss"]) * 1.1])
    plt.legend()
    plt.grid(True)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/training_history_{model_type}.png", dpi=500)
    plt.close()

    loss_data = {
        model_type: {
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
        }
    }
    log_path = f"{folder}/loss_history.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(loss_data)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=4)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_model_cgtst(
    cgtst: CGTST,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    lr: float,
    max_iter: int,
    lam: float = 0.0,
    lam_ridge: float = 0.0,
    check_every: int = 1,
    verbose: int = 1,
) -> dict:
    """
    Train the CGTST model using the Adam optimiser with optional L1 and ridge
    regularisation on the causality gates.

    Args:
        cgtst (CGTST): Model to train.
        X_train, Y_train: Training tensors.
        X_val, Y_val: Validation tensors.
        lr (float): Learning rate.
        max_iter (int): Number of epochs.
        lam (float): L1 sparsity penalty on causality gates (λ_K in paper).
        lam_ridge (float): Ridge penalty on Transformer weights (λ_M in paper).
        check_every (int): Logging frequency (epochs).
        verbose (int): Verbosity level.

    Returns:
        dict: {'train_loss': [...], 'val_loss': [...]}
    """
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(cgtst.parameters(), lr=lr)
    history: dict = {"train_loss": [], "val_loss": []}

    for it in range(max_iter):
        cgtst.train()
        y_pred = cgtst(X_train)
        loss = loss_fn(y_pred, Y_train)

        reg = 0.0
        if lam > 0:
            reg += regularize(cgtst, lam)
        if lam_ridge > 0:
            reg += ridge_regularize(cgtst, lam_ridge)
        loss = loss + reg

        loss.backward()
        optimizer.step()
        cgtst.zero_grad()

        if (it + 1) % check_every == 0:
            cgtst.eval()
            with torch.no_grad():
                val_loss = loss_fn(cgtst(X_val), Y_val)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())

            if verbose > 0:
                print(f"{'—' * 10} Iter {it + 1} {'—' * 10}")
                print(f"  Train Loss = {loss.item():.6f}")
                print(f"  Val   Loss = {val_loss.item():.6f}")

    return history


# ---------------------------------------------------------------------------
# Permutation Feature Importance
# ---------------------------------------------------------------------------

def permutation_feature_importance(
    model: CGTST,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    selected_columns: list,
    threshold: float = 1.0,
) -> dict:
    """
    Permutation Feature Importance (PFI) to validate which features are causal.

    A PFI ratio < *threshold* indicates the feature carries causal information:
    permuting it raises the validation loss proportionally.

    Args:
        model (CGTST): Trained model.
        X_val (torch.Tensor): Validation input of shape (T, context, n_features).
        Y_val (torch.Tensor): Validation target.
        selected_columns (list): Feature names corresponding to the last dim of X_val.
        threshold (float): Features with PFI ratio < threshold are retained.

    Returns:
        dict: {col: pfi_ratio} for columns that pass the threshold.
    """
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    base_loss = loss_fn(model(X_val), Y_val).item()
    print(f"PFI — Baseline validation loss: {base_loss:.6f}")

    causal = {}
    for i, col in enumerate(selected_columns):
        X_perm = X_val.clone()
        perm_idx = torch.randperm(X_perm.size(1))
        X_perm[:, :, i] = X_perm[:, perm_idx, i]

        with torch.no_grad():
            perm_loss = loss_fn(model(X_perm), Y_val).item()

        ratio = base_loss / perm_loss if perm_loss != 0 else float("inf")
        print(f"  PFI ratio [{col}]: {ratio:.6f}")
        if ratio < threshold:
            causal[col] = ratio

    return causal


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def prepare_data_for_model_cv(
    data,
    selected_columns: list,
    clstm_params: dict,
    training_params: dict,
    folder: str,
    device: torch.device,
    n_splits: int = 5,
    min_train_ratio: float = 0.5,
    max_train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    case: str = "cgtst",
) -> tuple:
    """
    Time-series cross-validation (expanding window) for CGTST.

    Trains one CGTST model per fold, evaluates on a hold-out window, and
    aggregates metrics.  Reproduces the TimeSeriesSplit scheme from the paper.

    Args:
        data (pd.DataFrame): Full dataset (features + target).
        selected_columns (list): Feature column names.
        clstm_params (dict): Model hyperparameters.
        training_params (dict): Training hyperparameters.
        folder (str): Output directory for logs and plots.
        device (torch.device): Compute device.
        n_splits (int): Number of CV folds.
        min_train_ratio (float): Fraction of data used in the first (smallest) fold.
        max_train_ratio (float): Fraction of data used in the last (largest) fold.
        test_ratio (float): Fraction of data used as test window per fold.
        case (str): Identifier written to the log file.

    Returns:
        tuple: (last_history, models, fold_summaries, validation_data)
    """
    import numpy as np

    X = data[selected_columns].values
    y = data["unique_num_plate_count"].values
    total = len(X)
    test_size = int(test_ratio * total)
    min_train = int(min_train_ratio * total)
    max_train = int(max_train_ratio * total)
    inc = (max_train - min_train) // max(1, n_splits - 1)

    splits = []
    for fold in range(n_splits):
        tr_end = min_train + fold * inc
        te_end = min(tr_end + test_size, total)
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))

    context = clstm_params.get("context", 6)
    os.makedirs(folder, exist_ok=True)

    fold_summaries, histories, models, validation_data = [], [], [], []
    total_time = 0.0
    last_history = {}

    with open(f"{folder}/cgtst_pytorch.txt", "w") as f:
        f.write(f"\n\n--- {case.capitalize()} Model Results ---\n")
        f.write(f"Total samples: {total}\n")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) < context + 1:
                msg = f"Fold {fold_idx+1} skipped: insufficient training samples."
                print(msg); f.write(msg + "\n")
                continue

            t0 = time.time()
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_t = torch.from_numpy(X_tr).float().to(device)
            Y_tr_t = torch.from_numpy(y_tr).float().to(device)
            X_te_t = torch.from_numpy(X_te).float().to(device)
            Y_te_t = torch.from_numpy(y_te).float().to(device)

            try:
                X_arr, Y_arr = arrange_input(X_tr_t, context)
                X_val_arr, Y_val_arr = arrange_input(X_te_t, context)
            except ValueError as e:
                msg = f"Fold {fold_idx+1} skipped: {e}"
                print(msg); f.write(msg + "\n")
                continue

            out_dim = Y_arr.shape[1] if Y_arr.ndim > 1 else 1
            model = CGTST(
                input_dim=X_tr.shape[1],
                model_dim=clstm_params.get("model_dim", 64),
                num_heads=clstm_params.get("num_heads", 4),
                num_layers=clstm_params.get("num_layers", 2),
                dropout=clstm_params.get("dropout", 0.1),
                output_dim=out_dim,
                context=context,
            ).to(device)

            history = train_model_cgtst(
                model, X_arr, Y_arr, X_val_arr, Y_val_arr,
                lr=training_params.get("learning_rate", 0.01),
                max_iter=training_params.get("epochs", 200),
                lam=training_params.get("lam", 0.0),
                lam_ridge=training_params.get("lam_ridge", 0.0),
                check_every=training_params.get("check_every", 1),
                verbose=training_params.get("verbose", 1),
            )
            histories.append(history)
            models.append(model)
            validation_data.append((X_val_arr, Y_val_arr))
            last_history = history

            model.eval()
            with torch.no_grad():
                preds = model(X_val_arr).cpu().numpy()
                y_true = Y_val_arr.cpu().numpy()

            if preds.shape != y_true.shape:
                msg = f"Fold {fold_idx+1}: shape mismatch {preds.shape} vs {y_true.shape}, skipped."
                print(msg); f.write(msg + "\n")
                continue

            mae = mean_absolute_error(y_true, preds)
            mse = mean_squared_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            elapsed = time.time() - t0
            total_time += elapsed

            summary = {
                "fold": fold_idx + 1, "train_size": len(train_idx), "test_size": len(test_idx),
                "mae": mae, "mse": mse, "r2": r2,
            }
            fold_summaries.append(summary)

            f.write(f"\nFold {fold_idx+1}: MAE={mae:.4f}  MSE={mse:.4f}  R2={r2:.4f}  "
                    f"Time={elapsed:.2f}s\n")
            gates = model.get_causality_gates()
            f.write(f"  Causality gates: {gates.tolist()}\n")

        if fold_summaries:
            avg_mae = np.mean([s["mae"] for s in fold_summaries])
            avg_mse = np.mean([s["mse"] for s in fold_summaries])
            avg_r2  = np.mean([s["r2"]  for s in fold_summaries])
            avg_t   = total_time / len(fold_summaries)
            f.write(f"\nAverage — MAE: {avg_mae:.4f}  MSE: {avg_mse:.4f}  "
                    f"R2: {avg_r2:.4f}  Time: {avg_t:.2f}s\n")
        else:
            f.write("\nNo valid folds.\n")

    if histories:
        plot_loss(last_history, model_type="cgtst_pytorch", folder=folder)

    return last_history, models, fold_summaries, validation_data
