"""
model.py
--------
GCT — Granger Causal Transformer for traffic forecasting.

Architecture (from paper Table, Section 3.3):
  0. Input           : (N_batch, num_variables)
  1. LSTM            : (N_batch, lstm_units=32)         — temporal encoder
  2. LayerNorm       : (N_batch, lstm_units)
  3. CausalAttention : (N_batch, num_variables)          — num_heads=4, key_dim=num_variables
                         Attention(Q,K,V) = softmax(QK^T/√d_k ⊙ (1+C)) V
  4. LayerNorm       : (N_batch, num_variables)
  5. Add (residual)  : adds dense_proj (projected LSTM) to attention output
  6. Concatenate     : (N_batch, lstm_units + num_variables)
  7. Dense(relu)     : (N_batch, dense_units=64)
  8. Dropout(0.15)
  9. Dense(relu)     : (N_batch, dense_units//2=32)
  10. Dropout(0.15)
  11. Dense(1)       : scalar forecast

The causal matrix C ∈ R^{m×m} encodes Granger relationships:
  C[i,j] = 1/best_lag  if X_j Granger-causes X_i (p < 0.05)
  C[i,j] = 0           otherwise
C is built by granger_analysis.create_causal_matrix(). See paper Eq. (4).

References
----------
Durán-López et al., "Forecasting Traffic in Smart Villages via Granger-Causal
Attention", ACM (2025). See paper/figures/BlockDiagram.png.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import load_and_prepare_data
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


# ---------------------------------------------------------------------------
# Causal Self-Attention  (paper Eq. 4)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention modified with a Granger causal mask.

    The attention score matrix is element-wise multiplied (Hadamard product)
    by (1 + C), where C is the Granger causal matrix:

        Attention(Q, K, V) = softmax(QK^T / √d_k  ⊙  (1 + C)) V

    This boosts attention scores between causally related pairs and leaves
    non-causal pairs unchanged (C[i,j]=0 → factor = 1).

    Args:
        num_variables (int): Number of input features m (= key_dim in paper).
        num_heads (int): Number of attention heads (paper: 4).
        causal_matrix (torch.Tensor | None): Shape (m, m).  If None, C = 0
            (reduces to standard attention — used for ablation).
        alpha (float): Scaling factor on C. Paper uses alpha=1.
    """

    def __init__(
        self,
        num_variables: int,
        num_heads: int = 4,
        causal_matrix: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.num_heads = num_heads
        self.alpha = alpha
        self.head_dim = max(1, num_variables // num_heads)

        self.q_proj = nn.Linear(num_variables, num_heads * self.head_dim)
        self.k_proj = nn.Linear(num_variables, num_heads * self.head_dim)
        self.v_proj = nn.Linear(num_variables, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, num_variables)

        # Register causal matrix as a non-trainable buffer
        if causal_matrix is not None:
            self.register_buffer("C", causal_matrix.float())
        else:
            self.register_buffer("C", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_variables)  or  (batch, num_variables)
        Returns:
            out: same shape as x
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)   # treat as seq_len=1
            squeeze = True

        B, T, m = x.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Scaled dot-product: (B, H, T, T)
        scale = d ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply causal mask ⊙ (1 + alpha*C) per paper Eq. (4)
        if self.C is not None:
            # C is (m, m). We use the first T×T block (T=1 for our tabular case).
            # In the tabular setting T=1, so scores is (B, H, 1, 1) — trivial.
            # For sequence inputs (T=m) the mask is (T, T).
            mask = 1.0 + self.alpha * self.C[:T, :T]   # (T, T)
            scores = scores * mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)               # (B, H, T, T)
        out = torch.matmul(attn, V)                    # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, H * d)
        out = self.out_proj(out)                       # (B, T, m)

        if squeeze:
            out = out.squeeze(1)
        return out


# ---------------------------------------------------------------------------
# GCT — Granger Causal Transformer  (paper Table, Section 3.3)
# ---------------------------------------------------------------------------

class GCT(nn.Module):
    """
    Granger Causal Transformer for multivariate traffic forecasting.

    Full architecture (paper Table, all hypers as published):
        Input (N_batch, num_variables)
        → LSTM(lstm_units=32, return_sequences=True)
        → LayerNorm
        → [Dense projection to num_variables]        # aligns dims for attention
        → CausalSelfAttention(num_heads=4, key_dim=num_variables, C=causal_matrix)
        → LayerNorm
        → Add (residual: dense_proj + attention)
        → Concatenate([LSTM output, Add output])
        → Dense(dense_units=64, ReLU)
        → Dropout(0.15)
        → Dense(dense_units//2=32, ReLU)
        → Dropout(0.15)
        → Dense(1)

    Args:
        num_variables (int): Number of input features m.
        lstm_units (int): LSTM hidden size (paper: 32).
        num_heads (int): Attention heads (paper: 4).
        dense_units (int): First dense layer width (paper: 64).
        dropout_rate (float): Dropout probability (paper: 0.15).
        causal_matrix (torch.Tensor | None): Pre-computed Granger matrix C.
        alpha (float): Causal mask scaling (paper: 1.0).
        use_causal_mask (bool): If False, C is zeroed — ablation switch.
    """

    def __init__(
        self,
        num_variables: int,
        lstm_units: int = 32,
        num_heads: int = 4,
        dense_units: int = 64,
        dropout_rate: float = 0.15,
        causal_matrix: torch.Tensor | None = None,
        alpha: float = 1.0,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.lstm_units = lstm_units

        # Step 1: LSTM sequential encoder
        self.lstm = nn.LSTM(
            input_size=num_variables,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(lstm_units)

        # Project LSTM output to num_variables for attention compatibility
        self.dense_proj = nn.Linear(lstm_units, num_variables, bias=False)

        # Step 2: Causal self-attention with Granger mask (paper Eq. 4)
        C = causal_matrix if use_causal_mask else None
        self.attention = CausalSelfAttention(
            num_variables=num_variables,
            num_heads=num_heads,
            causal_matrix=C,
            alpha=alpha,
        )
        self.ln2 = nn.LayerNorm(num_variables)

        # Step 3: Feed-forward classifier (Concatenate → Dense → Dense → Dense(1))
        concat_dim = lstm_units + num_variables
        self.ff = nn.Sequential(
            nn.Linear(concat_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_variables)
        Returns:
            y: (batch,) — scalar forecast
        """
        # LSTM: temporal encoding → take last hidden state
        lstm_out, _ = self.lstm(x)          # (B, T, lstm_units)
        lstm_out = self.ln1(lstm_out)        # LayerNorm
        lstm_last = lstm_out[:, -1, :]       # (B, lstm_units)

        # Project to attention space and apply causal self-attention
        proj = self.dense_proj(lstm_out)     # (B, T, num_variables)
        attn = self.attention(proj)          # (B, T, num_variables)
        attn = self.ln2(attn)
        attn_last = attn[:, -1, :]           # (B, num_variables)

        # Residual + Concatenate
        added = proj[:, -1, :] + attn_last   # (B, num_variables)
        concat = torch.cat([lstm_last, added], dim=-1)  # (B, lstm_units + num_variables)

        return self.ff(concat).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Regularisation helpers (kept for backward compatibility / future use)
# ---------------------------------------------------------------------------

def regularize(model: nn.Module, lam: float) -> torch.Tensor:
    """
    L1 regularisation on all model parameters.

    Args:
        model: Any nn.Module.
        lam: Regularisation coefficient.
    """
    return lam * sum(p.abs().sum() for p in model.parameters())


def ridge_regularize(model: nn.Module, lam: float) -> torch.Tensor:
    """
    Ridge (L2) regularisation on all model parameters.

    Args:
        model: Any nn.Module.
        lam: Regularisation coefficient.
    """
    return lam * sum((p ** 2).sum() for p in model.parameters())


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
        target_seq: (T - context, 1)  — the step immediately after each window
    """
    assert context >= 1 and isinstance(context, int)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D tensor of shape (T, dim).")
    T, dim = data.shape
    input_seq = torch.zeros(T - context, context, dim, dtype=torch.float32, device=data.device)
    for i in range(context):
        input_seq[:, i, :] = data[i: T - context + i]
    target_seq = data[context:T, -1:]   # target is the last column (plate count)
    return input_seq.detach(), target_seq.detach()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GCT — Granger Causal Transformer")
    parser.add_argument("--seed", type=int, default=161,
                        help="Random seed (paper uses 161)")
    parser.add_argument("--folder", type=str, default="plots161")
    parser.add_argument("--trends_file", type=str, default="./DB/google_trends_data.csv")
    parser.add_argument("--plates_file", type=str, default="./DB/unique_num_plate_count_per_week.csv")
    # Paper hyperparameters (Table, Section 3.3)
    parser.add_argument("--lstm_units", type=int, default=32,
                        help="LSTM hidden units (paper: 32)")
    parser.add_argument("--dense_units", type=int, default=64,
                        help="First dense layer width (paper: 64)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Attention heads (paper: 4)")
    parser.add_argument("--dropout_rate", type=float, default=0.15,
                        help="Dropout probability (paper: 0.15)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Adam learning rate (paper: 0.0001)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (paper: 200)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (paper: N_batch=64)")
    parser.add_argument("--input_window", type=int, default=6,
                        help="Input context window (paper: 6 weeks)")
    parser.add_argument("--max_lag", type=int, default=12,
                        help="Maximum Granger lag (paper: p_max=12)")
    parser.add_argument("--corr_threshold", type=float, default=0.5,
                        help="Correlation filter threshold alpha (paper: 0.5)")
    parser.add_argument("--alpha_causal", type=float, default=1.0,
                        help="Causal mask scaling (paper: 1.0)")
    # Cross-validation (paper: 10-fold expanding window)
    parser.add_argument("--n_splits", type=int, default=10,
                        help="Number of CV folds (paper: 10)")
    parser.add_argument("--min_train_ratio", type=float, default=0.5)
    parser.add_argument("--max_train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    from .granger_analysis import granger_causality_test, generate_shifted_trends, create_causal_matrix

    data = load_and_prepare_data(
        args.trends_file, args.plates_file,
        normalize=True, apply_eemd=True, num_imfs=5, noise_width=0.05,
    )

    # Feature selection: Granger causality + correlation filter
    results, lags, valid_lags, _ = granger_causality_test(data, max_lag=args.max_lag)
    selected_trends = [t for t, p in results.items() if p < 0.05]
    print(f"Selected trends (p < 0.05): {selected_trends}")

    all_trends = generate_shifted_trends(data, valid_lags)
    correlations = all_trends.corr()["unique_num_plate_count"]
    selected_columns = [
        col for col in all_trends.columns
        if abs(correlations.get(col, 0)) >= args.corr_threshold
        and any(t in col for t in selected_trends)
    ]
    print(f"Final feature columns ({len(selected_columns)}): {selected_columns}")

    # Build Granger causal matrix C
    causal_matrix_np = create_causal_matrix(data, selected_trends, lags, max_lag=args.max_lag)
    causal_matrix = torch.tensor(causal_matrix_np, dtype=torch.float32)

    model_params = {
        "lstm_units": args.lstm_units,
        "dense_units": args.dense_units,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "alpha_causal": args.alpha_causal,
    }
    training_params = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
    }

    print("Starting GCT cross-validation training (10-fold expanding window)...")
    last_history, models, fold_summaries, validation_data = prepare_data_for_model_cv(
        data=all_trends,
        selected_columns=selected_columns,
        model_params=model_params,
        training_params=training_params,
        causal_matrix=causal_matrix,
        folder=args.folder,
        device=device,
        n_splits=args.n_splits,
        min_train_ratio=args.min_train_ratio,
        max_train_ratio=args.max_train_ratio,
        test_ratio=args.test_ratio,
        context=args.input_window,
        case="gct_full",
    )

    print("\nRunning Permutation Feature Importance (PFI)...")
    for idx, model in enumerate(models):
        X_val_arr, Y_val_arr = validation_data[idx]
        causal = permutation_feature_importance(
            model, X_val_arr, Y_val_arr, selected_columns, threshold=1.0
        )
        print(f"  Fold {idx + 1} causal features: {causal}")

    if last_history:
        plot_loss(last_history, model_type="gct", folder=args.folder)
    print("Done.")


if __name__ == "__main__":
    main()
