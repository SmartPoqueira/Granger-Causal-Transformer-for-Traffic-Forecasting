"""
training.py
-----------
Cross-validation training loop, metrics logging, and loss-curve plotting
for the GCT (Granger Causal Transformer) pipeline.

All hyperparameters match the published paper values (Section 3.3, Table 1):
  - optimizer: Adam, lr=0.0001
  - epochs: 200
  - batch_size: 64  (N_batch in paper)
  - n_splits: 10 (expanding window CV)
  - dropout: 0.15
"""

import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss(history: dict, model_type: str = "gct", folder: str = "plots") -> None:
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
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/training_history_{model_type}.png", dpi=500)
    plt.close()

    log_path = f"{folder}/loss_history.json"
    existing = {}
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            existing = json.load(f)
    existing[model_type] = {
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
    }
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=4)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_model_gct(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    lr: float = 0.0001,
    max_iter: int = 200,
    batch_size: int = 64,
    verbose: int = 0,
) -> dict:
    """
    Train the GCT model using Adam optimiser + MSE loss.

    Paper hyperparameters (Table, Section 3.3):
        lr=0.0001, epochs=200, batch_size=64 (N_batch).

    Args:
        model: GCT instance.
        X_train: (N, context, num_variables)
        Y_train: (N, 1)
        X_val:   (M, context, num_variables)
        Y_val:   (M, 1)
        lr: Adam learning rate.
        max_iter: Number of training epochs.
        batch_size: Mini-batch size.
        verbose: Print frequency (0 = silent).

    Returns:
        dict: {'train_loss': [...], 'val_loss': [...]}
    """
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict = {"train_loss": [], "val_loss": []}
    N = X_train.shape[0]

    for epoch in range(max_iter):
        model.train()
        # Mini-batch loop
        perm = torch.randperm(N, device=X_train.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
            x_b = X_train[idx]
            y_b = Y_train[idx].squeeze(-1)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val.squeeze(-1))

        avg_train = epoch_loss / max(1, n_batches)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss.item())

        if verbose > 0 and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:03d}/{max_iter}  "
                  f"train_loss={avg_train:.6f}  val_loss={val_loss.item():.6f}")

    return history


# ---------------------------------------------------------------------------
# Permutation Feature Importance
# ---------------------------------------------------------------------------

def permutation_feature_importance(
    model: nn.Module,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    selected_columns: list,
    threshold: float = 1.0,
) -> dict:
    """
    Permutation Feature Importance (PFI) to validate causal features.

    A PFI ratio < *threshold* indicates the feature carries predictive signal:
    permuting it raises the validation loss proportionally.

    Args:
        model: Trained GCT.
        X_val: (T, context, n_features)
        Y_val: (T, 1)
        selected_columns: Feature names.
        threshold: Ratio cutoff.

    Returns:
        dict: {col: pfi_ratio} for columns that pass.
    """
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        base_loss = loss_fn(model(X_val), Y_val.squeeze(-1)).item()
    print(f"PFI — Baseline validation loss: {base_loss:.6f}")

    causal = {}
    for i, col in enumerate(selected_columns):
        X_perm = X_val.clone()
        perm_idx = torch.randperm(X_perm.size(0))
        X_perm[:, :, i] = X_perm[perm_idx, :, i]
        with torch.no_grad():
            perm_loss = loss_fn(model(X_perm), Y_val.squeeze(-1)).item()
        ratio = base_loss / perm_loss if perm_loss > 0 else float("inf")
        print(f"  PFI [{col}]: {ratio:.4f}")
        if ratio < threshold:
            causal[col] = ratio
    return causal


# ---------------------------------------------------------------------------
# Cross-validation loop  (10-fold expanding window, paper Section 4)
# ---------------------------------------------------------------------------

def prepare_data_for_model_cv(
    data,
    selected_columns: list,
    model_params: dict,
    training_params: dict,
    causal_matrix: torch.Tensor | None,
    folder: str,
    device: torch.device,
    n_splits: int = 10,
    min_train_ratio: float = 0.5,
    max_train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    context: int = 6,
    case: str = "gct_full",
    use_causal_mask: bool = True,
) -> tuple:
    """
    10-fold expanding-window cross-validation for GCT.

    Paper uses:
        n_splits=10, min_train_ratio=0.5, max_train_ratio=0.8, test_ratio=0.2,
        context (input window) = 6 weeks.

    Args:
        data: pd.DataFrame with features + 'unique_num_plate_count' target.
        selected_columns: Feature columns after Granger + correlation filtering.
        model_params: Dict with lstm_units, dense_units, num_heads,
                      dropout_rate, alpha_causal.
        training_params: Dict with learning_rate, epochs, batch_size, verbose.
        causal_matrix: Precomputed Granger C matrix (num_features × num_features).
        folder: Output directory.
        device: Torch device.
        n_splits, min_train_ratio, max_train_ratio, test_ratio: CV scheme.
        context: Sliding window length (paper: 6).
        case: Log filename identifier.
        use_causal_mask: If False, C is disabled (ablation switch).

    Returns:
        tuple: (last_history, models, fold_summaries, validation_data)
    """
    from .model import GCT, arrange_input, regularize, ridge_regularize

    # Build data arrays
    all_cols = selected_columns + (
        ["unique_num_plate_count"] if "unique_num_plate_count" not in selected_columns else []
    )
    X_full = data[all_cols].values.astype(np.float32)
    total = len(X_full)
    test_size = int(test_ratio * total)
    min_train = int(min_train_ratio * total)
    max_train = int(max_train_ratio * total)
    inc = (max_train - min_train) // max(1, n_splits - 1)

    splits = []
    for fold in range(n_splits):
        tr_end = min_train + fold * inc
        te_end = min(tr_end + test_size, total)
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))

    num_variables = len(selected_columns) + 1   # features + target

    # Move causal matrix to device (sub-matrix for selected features only)
    C_device = None
    if causal_matrix is not None and use_causal_mask:
        C_device = causal_matrix.to(device)

    os.makedirs(folder, exist_ok=True)
    fold_summaries, histories, models, validation_data = [], [], [], []
    total_time = 0.0
    last_history = {}

    with open(f"{folder}/{case}.txt", "w") as f:
        f.write(f"\n\n--- {case} Results (n_splits={n_splits}, context={context}) ---\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"use_causal_mask: {use_causal_mask}\n\n")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) < context + 1 or len(test_idx) < context + 1:
                msg = f"Fold {fold_idx+1} skipped: insufficient samples."
                print(msg); f.write(msg + "\n")
                continue

            t0 = time.time()
            X_tr = torch.from_numpy(X_full[train_idx]).to(device)
            X_te = torch.from_numpy(X_full[test_idx]).to(device)

            try:
                X_arr, Y_arr = arrange_input(X_tr, context)
                X_val_arr, Y_val_arr = arrange_input(X_te, context)
            except ValueError as e:
                msg = f"Fold {fold_idx+1} skipped: {e}"
                print(msg); f.write(msg + "\n")
                continue

            # Instantiate GCT with paper hyperparameters
            model = GCT(
                num_variables=num_variables,
                lstm_units=model_params.get("lstm_units", 32),
                num_heads=model_params.get("num_heads", 4),
                dense_units=model_params.get("dense_units", 64),
                dropout_rate=model_params.get("dropout_rate", 0.15),
                causal_matrix=C_device,
                alpha=model_params.get("alpha_causal", 1.0),
                use_causal_mask=use_causal_mask,
            ).to(device)

            history = train_model_gct(
                model=model,
                X_train=X_arr,
                Y_train=Y_arr,
                X_val=X_val_arr,
                Y_val=Y_val_arr,
                lr=training_params.get("learning_rate", 0.0001),
                max_iter=training_params.get("epochs", 200),
                batch_size=training_params.get("batch_size", 64),
                verbose=training_params.get("verbose", 0),
            )
            histories.append(history)
            models.append(model)
            validation_data.append((X_val_arr, Y_val_arr))
            last_history = history

            model.eval()
            with torch.no_grad():
                preds = model(X_val_arr).cpu().numpy()
                y_true = Y_val_arr.squeeze(-1).cpu().numpy()

            mae = mean_absolute_error(y_true, preds)
            mse = mean_squared_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            elapsed = time.time() - t0
            total_time += elapsed

            fold_summaries.append({
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "mae": mae, "mse": mse, "r2": r2,
            })
            msg = (f"Fold {fold_idx+1}: MAE={mae:.4f}  MSE={mse:.4f}  "
                   f"R²={r2:.4f}  Time={elapsed:.2f}s")
            print(f"  {msg}")
            f.write(msg + "\n")

        if fold_summaries:
            avg_mae = np.mean([s["mae"] for s in fold_summaries])
            avg_mse = np.mean([s["mse"] for s in fold_summaries])
            avg_r2  = np.mean([s["r2"]  for s in fold_summaries])
            std_mae = np.std([s["mae"] for s in fold_summaries])
            std_mse = np.std([s["mse"] for s in fold_summaries])
            std_r2  = np.std([s["r2"]  for s in fold_summaries])
            avg_t   = total_time / len(fold_summaries)
            summary_line = (f"\nAverage — MAE: {avg_mae:.4f}±{std_mae:.4f}  "
                            f"MSE: {avg_mse:.4f}±{std_mse:.4f}  "
                            f"R²: {avg_r2:.4f}±{std_r2:.4f}  "
                            f"Time: {avg_t:.2f}s")
            print(summary_line)
            f.write(summary_line + "\n")
        else:
            f.write("\nNo valid folds.\n")

    if histories:
        plot_loss(last_history, model_type=case, folder=folder)

    return last_history, models, fold_summaries, validation_data
