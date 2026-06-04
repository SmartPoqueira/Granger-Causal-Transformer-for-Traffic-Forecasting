"""
granger_analysis.py
-------------------
Alternative Keras/TensorFlow model: an LSTM + self-attention architecture with
a Granger-causal masking layer.  This module implements the TF-based comparison
model used in the ablation study alongside the main PyTorch CGTST.

References
----------
Paper, Section 3.2 — "Causality-guided LSTM-Transformer (Keras baseline)".
"""

import os
import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.sandbox.stats.runs import runstest_1samp

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, Concatenate, Add, LSTM
)
from tensorflow.keras.optimizers import Adam

# Disable GPU for TF (runs alongside PyTorch experiments on the same machine)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---------------------------------------------------------------------------
# Seed & folder are read from CLI arguments when this script is run directly
# ---------------------------------------------------------------------------
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 11
FOLDER = sys.argv[2] if len(sys.argv) > 2 else "plots"


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# ---------------------------------------------------------------------------
# Data utilities (standalone — no dependency on PyTorch preprocessing module)
# ---------------------------------------------------------------------------

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise all columns to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    normalised = scaler.fit_transform(data)
    return pd.DataFrame(normalised, index=data.index, columns=data.columns)


def load_and_prepare_data(trends_file: str, plates_file: str, normalize: bool = True) -> pd.DataFrame:
    """Load and merge Google Trends and weekly plate-count CSVs."""
    trends = pd.read_csv(trends_file, parse_dates=["Semana"])
    plates = pd.read_csv(plates_file, parse_dates=["week"])
    data = pd.merge(trends, plates, left_on="Semana", right_on="week")
    data.set_index("Semana", inplace=True)
    data.drop(columns=["week"], inplace=True)
    if normalize:
        data = normalize_data(data)
    return data


def run_test(series: np.ndarray) -> float:
    """Return the p-value of a Wald–Wolfowitz runs test."""
    _, p_value = runstest_1samp(series)
    return p_value


def granger_causality_test(data: pd.DataFrame, max_lag: int = 6) -> tuple:
    """
    Run Granger causality tests for each feature against ``unique_num_plate_count``.

    Uses the F-test (ssr_ftest) p-value.  Valid lags are those with p < 0.05.

    Returns:
        tuple: (results, lags, valid_lags, run_test_results)
    """
    results, lags, valid_lags, run_test_results = {}, {}, {}, {}
    target = "unique_num_plate_count"

    for col in data.columns:
        if col == target:
            continue
        test_result = grangercausalitytests(data[[target, col]], max_lag, verbose=False)
        p_values = [test_result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        min_p = min(p_values)
        best_lag = p_values.index(min_p) + 1

        results[col] = min_p
        lags[col] = best_lag
        # enumerate starting from 1 to get correct lag indices
        valid_lags[col] = [lag for lag, p in enumerate(p_values, 1) if p < 0.05]
        run_test_results[col] = run_test(data[col].values)

    return results, lags, valid_lags, run_test_results


def generate_shifted_trends(data: pd.DataFrame, valid_lags: dict) -> pd.DataFrame:
    """Append lagged columns for every Granger-significant feature."""
    shifted = pd.DataFrame(index=data.index)
    for trend, lag_list in valid_lags.items():
        for lag in lag_list:
            shifted[f"{trend}_lag_{lag}"] = data[trend].shift(lag)
    return pd.concat([data, shifted], axis=1).dropna()


# ---------------------------------------------------------------------------
# Loss plotting
# ---------------------------------------------------------------------------

def plot_loss(history, model_type: str = "multivariate_causality") -> None:
    """Save training/validation loss curves from a Keras History object."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Train and Validation Loss per Epoch — {model_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([0, 0.5])
    plt.legend()
    plt.grid(True)
    os.makedirs(FOLDER, exist_ok=True)
    plt.savefig(f"{FOLDER}/training_history_{model_type}.png", dpi=500)
    plt.close()

    loss_data = {
        model_type: {
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
    }
    log_path = f"{FOLDER}/loss_history.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(loss_data)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=4)


# ---------------------------------------------------------------------------
# Causal self-attention layer (Keras)
# ---------------------------------------------------------------------------

class SelfAttentionWithCausality(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer modulated by a Granger causal matrix.

    The causal adjustment is computed as a Hadamard product:
        adjusted = attention_output * (1 + α · (X W_c))
    where W_c is the causal matrix and α is a scaling factor.

    When α = 0 the layer reduces to standard multi-head self-attention.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Projection dimension per head.
        causal_matrix: Tensor of shape (num_trends, num_trends).
        alpha (float): Causal adjustment scaling factor.
    """

    def __init__(self, num_heads: int, key_dim: int, causal_matrix, alpha: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.causal_matrix = causal_matrix
        self.alpha = alpha
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        if self.alpha != 0:
            causal_adjustment = tf.matmul(inputs, self.causal_matrix)
            causal_adjustment = 1 + self.alpha * causal_adjustment
            return attention_output * causal_adjustment
        return attention_output


# ---------------------------------------------------------------------------
# Causal matrix construction
# ---------------------------------------------------------------------------

def create_causal_matrix(data: pd.DataFrame, selected_trends: list, lags: dict, max_lag: int = 6) -> np.ndarray:
    """
    Build an (n_trends × n_trends) Granger causal weight matrix.

    Entry [i, j] = 1 / best_lag if trend_i Granger-causes trend_j at p < 0.05,
    otherwise 0.  The inverse-lag weighting reflects that shorter lags indicate
    stronger temporal causality.

    Args:
        data (pd.DataFrame): Dataset used for the Granger tests.
        selected_trends (list): Feature names (rows/cols of the matrix).
        lags (dict): {trend: best_lag} from granger_causality_test.
        max_lag (int): Maximum lag to test.

    Returns:
        np.ndarray: Square causal matrix of shape (n_trends, n_trends).
    """
    n = len(selected_trends)
    causal_matrix = np.zeros((n, n))

    for i, trend_i in enumerate(selected_trends):
        for j, trend_j in enumerate(selected_trends):
            if i == j:
                continue
            result = grangercausalitytests(data[[trend_j, trend_i]], max_lag, verbose=False)
            p_values = [result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
            min_p = min(p_values)
            if min_p < 0.05:
                best_lag = p_values.index(min_p) + 1
                causal_matrix[i, j] = 1.0 / best_lag

    return causal_matrix


# ---------------------------------------------------------------------------
# Keras model factory
# ---------------------------------------------------------------------------

def create_transformer_model(input_shape, params: dict, causal_matrix, alpha: float):
    """
    Build the Keras LSTM + causal self-attention model.

    Architecture:
        Input → LSTM → LayerNorm → Dense(proj) →
        SelfAttentionWithCausality → LayerNorm → Add → Concatenate →
        Dense(relu) → Dropout → Dense(relu) → Dropout → Dense(1)

    Args:
        input_shape (tuple): (timesteps, n_features).
        params (dict): Hyperparameters (lstm_units, dropout_rate, learning_rate, dense_units, epochs).
        causal_matrix: Numpy array passed to SelfAttentionWithCausality.
        alpha (float): Causal adjustment scaling factor.

    Returns:
        tf.keras.Model: Compiled model.
    """
    lstm_units = params["lstm_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    dense_units = params["dense_units"]
    num_trends = causal_matrix.shape[0]

    inputs = Input(shape=input_shape)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = LayerNormalization(epsilon=1e-6)(lstm_out)

    dense_proj = Dense(num_trends, use_bias=False)(lstm_out)
    attention = SelfAttentionWithCausality(
        num_heads=4, key_dim=num_trends, causal_matrix=causal_matrix, alpha=alpha
    )(dense_proj)
    attention = LayerNormalization(epsilon=1e-6)(attention)

    added = Add()([dense_proj, attention])
    concat = Concatenate()([inputs, added])

    out = Dense(dense_units, activation="relu")(concat)
    out = Dropout(dropout_rate)(out)
    out = Dense(dense_units // 2, activation="relu")(out)
    out = Dropout(dropout_rate)(out)
    outputs = Dense(1)(out)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# ---------------------------------------------------------------------------
# Cross-validation loop (Keras model)
# ---------------------------------------------------------------------------

def prepare_data_for_model_cv(
    data: pd.DataFrame,
    selected_columns: list,
    params: dict,
    causal_matrix,
    alpha: float,
    n_splits: int = 5,
    min_train_ratio: float = 0.5,
    max_train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    case: str = "multivariate_causality",
):
    """
    Expanding-window cross-validation for the Keras LSTM-attention model.

    Args:
        data (pd.DataFrame): Full dataset.
        selected_columns (list): Feature columns (excluding target).
        params (dict): Model hyperparameters.
        causal_matrix: Granger causal weight matrix.
        alpha (float): Causal adjustment scaling.
        n_splits, min_train_ratio, max_train_ratio, test_ratio: CV parameters.
        case (str): Log file identifier.

    Returns:
        Keras History object from the last fold.
    """
    X = data[selected_columns + ["unique_num_plate_count"]].values
    y = data["unique_num_plate_count"].values
    total = len(X)
    test_size = int(test_ratio * total)
    min_train = int(min_train_ratio * total)
    max_train = int(max_train_ratio * total)
    inc = (max_train - min_train) // max(1, n_splits - 1)

    splits = []
    for fold in range(n_splits):
        tr_end = min_train + fold * inc
        te_end = tr_end + test_size
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))

    fold_summaries, histories = [], []
    total_time = 0.0

    os.makedirs(FOLDER, exist_ok=True)
    with open(f"{FOLDER}/multivariate_causality.txt", "w") as f:
        f.write(f"\n\n--- {case.capitalize()} Model Results ---\n")
        f.write(f"Total samples: {total}\n")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            t0 = time.time()
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
            X_te = X_te.reshape(X_te.shape[0], 1, X_te.shape[1])

            model = create_transformer_model(
                (X_tr.shape[1], X_tr.shape[2]), params, causal_matrix, alpha
            )

            try:
                history = model.fit(
                    X_tr, y_tr.reshape(-1, 1),
                    epochs=params["epochs"],
                    validation_data=(X_te, y_te.reshape(-1, 1)),
                    batch_size=16,
                    verbose=0,
                )
                histories.append(history)
            except Exception as e:
                f.write(f"Fold {fold_idx+1} error during training: {e}\n")
                continue

            preds = model.predict(X_te).reshape(-1)
            y_te_flat = y_te.reshape(-1)
            mae = mean_absolute_error(y_te_flat, preds)
            mse = mean_squared_error(y_te_flat, preds)
            r2 = r2_score(y_te_flat, preds)
            elapsed = time.time() - t0
            total_time += elapsed

            fold_summaries.append({"fold": fold_idx + 1, "mae": mae, "mse": mse, "r2": r2})
            f.write(f"\nFold {fold_idx+1}: MAE={mae:.4f}  MSE={mse:.4f}  R2={r2:.4f}  Time={elapsed:.2f}s\n")

        if fold_summaries:
            avg_mae = np.mean([s["mae"] for s in fold_summaries])
            avg_mse = np.mean([s["mse"] for s in fold_summaries])
            avg_r2  = np.mean([s["r2"]  for s in fold_summaries])
            avg_t   = total_time / n_splits
            f.write(f"\nAverage — MAE: {avg_mae:.4f}  MSE: {avg_mse:.4f}  "
                    f"R2: {avg_r2:.4f}  Time: {avg_t:.2f}s\n")
        else:
            f.write("\nNo valid folds.\n")

    return histories[-1] if histories else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    trends_file = "./DB/google_trends_data.csv"
    plates_file = "./DB/unique_num_plate_count_per_week.csv"

    data = load_and_prepare_data(trends_file, plates_file, normalize=True)
    results, lags, valid_lags, _ = granger_causality_test(data)

    selected_trends = [t for t, p in results.items() if p < 0.05]
    print(f"Selected trends (p < 0.05): {selected_trends}")

    all_trends = generate_shifted_trends(data, valid_lags)
    correlations = all_trends.corr()["unique_num_plate_count"]
    selected_columns = [
        col for col in all_trends.columns
        if correlations[col] >= 0.5 and any(t in col for t in selected_trends)
    ]
    print(f"Feature columns for multivariate model: {selected_columns}")

    causal_matrix = create_causal_matrix(data, selected_trends, lags)
    alpha = 1

    transformer_params = {
        "lstm_units": 32,
        "num_heads": 4,
        "key_dim": 32,
        "dropout_rate": 0.15,
        "learning_rate": 0.0001,
        "dense_units": 64,
        "epochs": 200,
    }

    last_history = prepare_data_for_model_cv(
        all_trends, selected_columns, transformer_params, causal_matrix, alpha,
        n_splits=10, case="multivariate_causality",
    )

    if last_history is not None:
        plot_loss(last_history)


if __name__ == "__main__":
    main()
