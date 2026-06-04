"""
preprocessing.py
----------------
Data loading, normalisation, EEMD denoising, and Granger-causality
feature-engineering routines for the CGTST pipeline.

Install dependencies before use:
    pip install PyEMD statsmodels scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.sandbox.stats.runs import runstest_1samp

# Install PyEMD before running: pip install PyEMD
from PyEMD import EEMD


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise all columns to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    normalised = scaler.fit_transform(data)
    return pd.DataFrame(normalised, index=data.index, columns=data.columns)


# ---------------------------------------------------------------------------
# EEMD denoising
# ---------------------------------------------------------------------------

def apply_eemd_denoising(data: pd.DataFrame, num_imfs: int = 5, noise_width: float = 0.05) -> pd.DataFrame:
    """
    Apply Ensemble Empirical Mode Decomposition (EEMD) to each column and
    reconstruct the signal by summing the lowest-frequency IMFs.

    Args:
        data (pd.DataFrame): Normalised data.
        num_imfs (int): Number of low-frequency IMFs to retain for reconstruction.
        noise_width (float): Standard deviation of Gaussian noise added per EEMD trial.

    Returns:
        pd.DataFrame: Denoised data with the same index and columns.
    """
    eemd = EEMD(trials=50, noise_width=noise_width)
    denoised = pd.DataFrame(index=data.index)

    for col in data.columns:
        signal = data[col].values
        imfs = eemd.eemd(signal)
        # Sum the first `num_imfs` IMFs (lowest frequency components)
        denoised[col] = np.sum(imfs[:num_imfs], axis=0)

    return denoised


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_data(
    trends_file: str,
    plates_file: str,
    normalize: bool = True,
    apply_eemd: bool = True,
    num_imfs: int = 5,
    noise_width: float = 0.05,
) -> pd.DataFrame:
    """
    Load and merge Google Trends and vehicle plate-count CSVs, then optionally
    normalise and apply EEMD denoising.

    Args:
        trends_file (str): Path to the Google Trends CSV (column 'Semana').
        plates_file (str): Path to the weekly plate-count CSV (column 'week').
        normalize (bool): Apply MinMax normalisation before EEMD.
        apply_eemd (bool): Apply EEMD denoising.
        num_imfs (int): Number of low-frequency IMFs to retain.
        noise_width (float): EEMD noise standard deviation.

    Returns:
        pd.DataFrame: Merged, normalised, and denoised dataset indexed by week.
    """
    trends = pd.read_csv(trends_file, parse_dates=["Semana"])
    plates = pd.read_csv(plates_file, parse_dates=["week"])

    data = pd.merge(trends, plates, left_on="Semana", right_on="week")
    data.set_index("Semana", inplace=True)
    data.drop(columns=["week"], inplace=True)

    if normalize:
        data = normalize_data(data)

    if apply_eemd:
        print("Applying Ensemble Empirical Mode Decomposition (EEMD) for denoising...")
        data = apply_eemd_denoising(data, num_imfs=num_imfs, noise_width=noise_width)
        print("EEMD denoising completed.")

    return data


# ---------------------------------------------------------------------------
# Granger causality & feature engineering
# ---------------------------------------------------------------------------

def run_test(series: np.ndarray) -> float:
    """Run a Wald–Wolfowitz runs test on *series* and return the p-value."""
    _, p_value = runstest_1samp(series)
    return p_value


def granger_causality_test(data: pd.DataFrame, max_lag: int = 12) -> tuple:
    """
    Run Granger causality tests for every feature against the target variable
    ``unique_num_plate_count``.

    The F-test p-value (``ssr_ftest``) is used to select the best lag.
    max_lag is set to 12 by default (paper: p_max=12 weeks, Section 3.2).

    Args:
        data (pd.DataFrame): DataFrame containing the target and all features.
        max_lag (int): Maximum lag order to test (paper: 12).

    Returns:
        tuple: (results, lags, valid_lags, run_test_results)
            - results (dict): {col: min_p_value}
            - lags (dict): {col: best_lag}
            - valid_lags (dict): {col: [lags with p < 0.05]}
            - run_test_results (dict): {col: runs-test p-value}
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
        valid_lags[col] = [lag for lag, p in enumerate(p_values, 1) if p < 0.05]
        run_test_results[col] = run_test(data[col].values)

    return results, lags, valid_lags, run_test_results


def generate_shifted_trends(data: pd.DataFrame, valid_lags: dict) -> pd.DataFrame:
    """
    Generate lagged versions of each Granger-significant feature.

    For each feature with valid lags, a new column ``{feature}_lag_{lag}``
    is appended.  Rows with NaN (introduced by shifting) are dropped.

    Args:
        data (pd.DataFrame): Original dataset.
        valid_lags (dict): {feature: [lag_1, lag_2, …]} from granger_causality_test.

    Returns:
        pd.DataFrame: Extended dataset with lagged features, NaN rows removed.
    """
    shifted = pd.DataFrame(index=data.index)
    for trend, lag_list in valid_lags.items():
        for lag in lag_list:
            shifted[f"{trend}_lag_{lag}"] = data[trend].shift(lag)
    return pd.concat([data, shifted], axis=1).dropna()
