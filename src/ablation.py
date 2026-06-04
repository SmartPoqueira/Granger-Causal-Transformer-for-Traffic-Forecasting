"""
ablation.py
-----------
Ablation study for GCT (Granger Causal Transformer).

Paper Section 4.4: Eight configurations progressively remove the three key
components — causality filtering, correlation filtering, causal attention mask.

Configurations (Table results_ablation):
  1. Full model           — all three components
  2. Without Causality    — no Granger filter; all lagged series pass
  3. Without Correlation  — no correlation filter; all Granger-significant lags pass
  4. Without Attention Mask — C = 0 (standard attention, no causal mask)
  5. Without Causality & Correlation
  6. Without Causality & Attention Mask
  7. Without Correlation & Attention Mask
  8. Base model           — none of the three components

Usage:
    python -m src.ablation --trends_file ./DB/google_trends_data.csv \\
                           --plates_file ./DB/unique_num_plate_count_per_week.csv \\
                           --folder ablation_results
"""

import argparse
import torch
import numpy as np

from .preprocessing import load_and_prepare_data
from .granger_analysis import granger_causality_test, generate_shifted_trends, create_causal_matrix
from .training import prepare_data_for_model_cv, plot_loss

# Paper hyperparameters (Section 3.3 / Table)
PAPER_MODEL_PARAMS = {
    "lstm_units":    32,
    "dense_units":   64,
    "num_heads":     4,
    "dropout_rate":  0.15,
    "alpha_causal":  1.0,
}
PAPER_TRAINING_PARAMS = {
    "learning_rate": 0.0001,
    "epochs":        200,
    "batch_size":    64,
    "verbose":       0,
}
# Cross-validation scheme (paper Section 4)
PAPER_CV = dict(n_splits=10, min_train_ratio=0.5, max_train_ratio=0.8, test_ratio=0.2, context=6)

# Granger parameters
MAX_LAG       = 12    # p_max (paper Section 3.2)
CORR_ALPHA    = 0.5   # best configuration from sensitivity analysis


def run_ablation(args):
    """Run all 8 ablation configurations and print a summary table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and preprocessing data...")
    data = load_and_prepare_data(
        args.trends_file, args.plates_file,
        normalize=True, apply_eemd=True, num_imfs=5, noise_width=0.05,
    )

    # Build base feature sets
    results, lags, valid_lags, _ = granger_causality_test(data, max_lag=MAX_LAG)
    all_lags = {}
    for col in data.columns:
        if col == "unique_num_plate_count":
            continue
        # All lags 1..MAX_LAG (no Granger filter)
        all_lags[col] = list(range(1, MAX_LAG + 1))

    # Granger-filtered lags
    granger_lags = valid_lags   # only lags with p < 0.05

    # Build feature DataFrames
    all_shifted     = generate_shifted_trends(data, all_lags)
    granger_shifted = generate_shifted_trends(data, granger_lags)

    # Correlation-filtered column sets
    def corr_filter(df, selected_trends):
        correlations = df.corr()["unique_num_plate_count"]
        return [
            c for c in df.columns
            if abs(correlations.get(c, 0)) >= CORR_ALPHA
            and any(t in c for t in selected_trends)
        ]

    granger_trends = [t for t, p in results.items() if p < 0.05]
    all_trends     = [c for c in data.columns if c != "unique_num_plate_count"]

    cols_full     = corr_filter(granger_shifted, granger_trends)   # Granger + Corr
    cols_no_corr  = [c for c in granger_shifted.columns            # Granger only
                     if any(t in c for t in granger_trends)]
    cols_no_gran  = corr_filter(all_shifted, all_trends)           # Corr only
    cols_base     = [c for c in all_shifted.columns                # Neither
                     if any(t in c for t in all_trends)]

    # Granger causal matrix C (used only when mask is active)
    causal_np = create_causal_matrix(data, granger_trends, lags, max_lag=MAX_LAG)
    causal_C  = torch.tensor(causal_np, dtype=torch.float32)

    # Eight ablation configurations (paper Table results_ablation)
    configs = [
        ("Full model",                        granger_shifted, cols_full,    causal_C, True),
        ("Without Causality",                 all_shifted,     cols_no_gran, causal_C, True),
        ("Without Correlation",               granger_shifted, cols_no_corr, causal_C, True),
        ("Without Attention Mask",            granger_shifted, cols_full,    causal_C, False),
        ("Without Causality & Correlation",   all_shifted,     cols_base,    causal_C, True),
        ("Without Causality & Att. Mask",     all_shifted,     cols_no_gran, causal_C, False),
        ("Without Correlation & Att. Mask",   granger_shifted, cols_no_corr, causal_C, False),
        ("Base model",                        all_shifted,     cols_base,    causal_C, False),
    ]

    print(f"\n{'='*70}")
    print(f"{'Configuration':<40} {'MAE':>8} {'MSE':>8} {'R²':>8} {'Time':>7}")
    print(f"{'='*70}")

    results_summary = []
    for name, df, cols, C, use_mask in configs:
        if not cols:
            print(f"{name:<40} {'(no features — skipped)':>30}")
            continue

        case_id = name.lower().replace(" ", "_").replace("&", "and")
        print(f"\nRunning: {name}")

        _, _, fold_summaries, _ = prepare_data_for_model_cv(
            data=df,
            selected_columns=cols,
            model_params=PAPER_MODEL_PARAMS,
            training_params=PAPER_TRAINING_PARAMS,
            causal_matrix=C,
            folder=f"{args.folder}/{case_id}",
            device=device,
            use_causal_mask=use_mask,
            case=case_id,
            **PAPER_CV,
        )

        if fold_summaries:
            avg_mae = np.mean([s["mae"] for s in fold_summaries])
            avg_mse = np.mean([s["mse"] for s in fold_summaries])
            avg_r2  = np.mean([s["r2"]  for s in fold_summaries])
            avg_t   = np.mean([s.get("time", 0) for s in fold_summaries])
            print(f"{name:<40} {avg_mae:>8.4f} {avg_mse:>8.4f} {avg_r2:>8.4f} {avg_t:>6.1f}s")
            results_summary.append((name, avg_mae, avg_mse, avg_r2))

    print(f"\n{'='*70}")
    print("Ablation complete. Results saved to:", args.folder)


def main():
    parser = argparse.ArgumentParser(description="GCT Ablation Study (paper Section 4.4)")
    parser.add_argument("--trends_file", type=str, default="./DB/google_trends_data.csv")
    parser.add_argument("--plates_file", type=str, default="./DB/unique_num_plate_count_per_week.csv")
    parser.add_argument("--folder", type=str, default="ablation_results",
                        help="Output directory for ablation results")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
