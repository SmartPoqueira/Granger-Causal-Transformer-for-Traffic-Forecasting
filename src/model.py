"""
model.py
--------
Neural network architectures for the CGTST (Causality-Gated Time Series
Transformer) pipeline.

References
----------
Durán-López et al., "Forecasting Traffic in Smart Villages via Granger-Causal
Attention", ACM (2025).  See paper/figures/BlockDiagram.png for the architecture.

Usage
-----
    from src.model import CGTST, arrange_input, regularize, ridge_regularize
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from .preprocessing import load_and_prepare_data, granger_causality_test, generate_shifted_trends
from .training import prepare_data_for_model_cv, permutation_feature_importance, plot_loss

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation_helper(name: str) -> nn.Module:
    """Return an activation module by name ('relu', 'tanh', or 'sigmoid')."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in Vaswani et al. (2017).

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model).
        Returns:
            x + positional encoding, same shape.
        """
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# TST — Time Series Transformer
# ---------------------------------------------------------------------------

class TST(nn.Module):
    """
    Time Series Transformer (TST) backbone.

    Projects input features to *model_dim*, applies positional encoding,
    passes through a stack of Transformer encoder layers, then flattens and
    linearly projects to *output_dim*.

    Args:
        input_dim (int): Number of input features (m in the paper).
        model_dim (int): Transformer hidden dimension (d).
        num_heads (int): Number of multi-head attention heads.
        num_layers (int): Number of Transformer encoder layers.
        dropout (float): Dropout rate.
        output_dim (int): Output dimension (n = 1 for single-step forecasting).
        context (int): Input window length (w).
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        output_dim: int = 1,
        context: int = 6,
    ) -> None:
        super().__init__()
        self.context = context
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.output_linear = nn.Linear(model_dim * context, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, w, m)
        Returns:
            y_pred: (batch, n)
        """
        x = self.input_linear(x)           # (batch, w, d)
        x = self.positional_encoding(x)    # (batch, w, d)
        x = self.transformer_encoder(x)    # (batch, w, d)
        x = self.flatten(x)                # (batch, w*d)
        return self.output_linear(x)       # (batch, n)


# ---------------------------------------------------------------------------
# CGTST — Causality-Gated Time Series Transformer
# ---------------------------------------------------------------------------

class CGTST(nn.Module):
    """
    Causality-Gated Time Series Transformer (CGTST).

    Wraps a TST with a learnable causality gate vector g ∈ R^m.
    At inference time, softmax(g) is element-wise multiplied with the input
    to weight features by their estimated causal relevance.

    The gate is regularised with L1 (λ_K) and ridge (λ_M) penalties to
    encourage sparsity (see paper, Eq. 3–5).

    Args:
        input_dim (int): Number of input features m.
        model_dim (int): Transformer hidden dimension d.
        num_heads (int): Multi-head attention heads.
        num_layers (int): Transformer encoder layers.
        dropout (float): Dropout rate.
        output_dim (int): Forecast horizon n (1 for one-step-ahead).
        context (int): Input window length w.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        output_dim: int = 1,
        context: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.causality_gate = nn.Parameter(torch.ones(input_dim))
        self.tst = TST(input_dim, model_dim, num_heads, num_layers, dropout, output_dim, context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, w, m)
        Returns:
            y_pred: (batch, n)
        """
        gate = torch.softmax(self.causality_gate, dim=0)  # (m,)
        x = x * gate.unsqueeze(0).unsqueeze(0)             # (batch, w, m)
        return self.tst(x)

    def get_causality_gates(self) -> np.ndarray:
        """Return the softmax-normalised gate weights as a NumPy array."""
        return torch.softmax(self.causality_gate, dim=0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------

def regularize(network: CGTST, lam: float) -> torch.Tensor:
    """
    L1 regularisation on the causality gate (λ_K in the paper).
    Encourages sparsity: unimportant features are pushed toward zero.

    Args:
        network (CGTST): Model instance.
        lam (float): Regularisation coefficient λ_K.
    """
    return lam * torch.sum(torch.abs(network.causality_gate))


def ridge_regularize(network: CGTST, lam: float) -> torch.Tensor:
    """
    Ridge (L2) regularisation on the Transformer weight matrices (λ_M).

    Penalises the input projection, output projection, and the first
    feed-forward layer of the first Transformer encoder block.

    Args:
        network (CGTST): Model instance.
        lam (float): Regularisation coefficient λ_M.
    """
    return lam * (
        torch.sum(network.tst.input_linear.weight ** 2)
        + torch.sum(network.tst.output_linear.weight ** 2)
        + torch.sum(network.tst.transformer_encoder.layers[0].linear1.weight ** 2)
        + torch.sum(network.tst.transformer_encoder.layers[0].linear2.weight ** 2)
    )


def restore_parameters(model: nn.Module, best_model: nn.Module) -> None:
    """Copy parameter values from *best_model* into *model* in-place."""
    for p, bp in zip(model.parameters(), best_model.parameters()):
        p.data = bp.data.clone()


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------

def arrange_input(data: torch.Tensor, context: int) -> tuple:
    """
    Slice a 2-D time series into overlapping windows of length *context*.

    Args:
        data: Tensor of shape (T, dim).
        context (int): Window length w ≥ 1.

    Returns:
        input_seq:  (T - context, context, dim)
        target_seq: (T - context, dim)  — the step immediately after each window
    """
    assert context >= 1 and isinstance(context, int)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D tensor of shape (T, dim).")
    T, dim = data.shape
    input_seq = torch.zeros(T - context, context, dim, dtype=torch.float32, device=data.device)
    for i in range(context):
        input_seq[:, i, :] = data[i : T - context + i]
    target_seq = data[context:T].detach()
    return input_seq.detach(), target_seq


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CGTST — Causality-Gated Time Series Transformer")
    parser.add_argument("--seed", type=int, default=161)
    parser.add_argument("--folder", type=str, default="plots161")
    parser.add_argument("--trends_file", type=str, default="./DB/google_trends_data.csv")
    parser.add_argument("--plates_file", type=str, default="./DB/unique_num_plate_count_per_week.csv")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--input_window", type=int, default=6)
    parser.add_argument("--lam", type=float, default=1e-3, help="L1 gate sparsity coefficient λ_K")
    parser.add_argument("--lam_ridge", type=float, default=1e-3, help="Ridge weight penalty λ_M")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--min_train_ratio", type=float, default=0.5)
    parser.add_argument("--max_train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--check_every", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    data = load_and_prepare_data(
        args.trends_file, args.plates_file,
        normalize=True, apply_eemd=True, num_imfs=5, noise_width=0.05,
    )

    feature_columns = [c for c in data.columns if c != "unique_num_plate_count"]
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

    clstm_params = {
        "context": args.input_window,
        "model_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }
    training_params = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "lam": args.lam,
        "lam_ridge": args.lam_ridge,
        "check_every": args.check_every,
        "verbose": args.verbose,
        "batch_size": args.batch_size,
    }

    print("Starting CGTST cross-validation training...")
    last_history, models, fold_summaries, validation_data = prepare_data_for_model_cv(
        data, feature_columns, clstm_params, training_params,
        folder=args.folder, device=device,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        max_train_ratio=args.max_train_ratio,
        test_ratio=args.test_ratio,
        case="cgtst",
    )

    print("\nRunning Permutation Feature Importance (PFI)...")
    for idx, model in enumerate(models):
        X_val_arr, Y_val_arr = validation_data[idx]
        causal = permutation_feature_importance(
            model, X_val_arr, Y_val_arr, feature_columns, threshold=1.0
        )
        print(f"  Fold {idx + 1} causal features: {causal}")

    if last_history:
        plot_loss(last_history, model_type="cgtst_pytorch", folder=args.folder)
    print("Done.")


if __name__ == "__main__":
    main()
