# Granger-Causal Transformer for Traffic Flow Forecasting in Smart Villages

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Journal: ACM TIST](https://img.shields.io/badge/Journal-ACM%20TIST-blue.svg)](https://dl.acm.org/doi/10.1145/3712290)

A time-series forecasting model that integrates **Granger causality analysis** with a **Transformer architecture** to predict traffic flow in smart villages. The model uses Google Trends data as exogenous signals, applying EEMD denoising and a Granger causal mask to weight attention scores based on statistically validated causal relationships between variables.

---

## Architecture

<p align="center">
  <img src="images/architecture.png" width="750"/>
</p>

*Full GCT pipeline: IoT data fusion with Google Trends → semantic validation (BETO/BERT) → EEMD denoising → Granger + correlation feature selection → causal matrix construction → LSTM encoder → Causality-Gated Transformer → forecast.*

---

## Overview

Predicting traffic flow in rural smart villages is challenging due to limited sensor infrastructure and strong seasonal variability. GCT addresses this by:

1. **Fusing heterogeneous data** — Combines IoT vehicle-detection counts with Google Trends search interest as exogenous variables.
2. **Semantic validation** — Uses a language model (BETO for Spanish, BERT for English) to retain only semantically relevant search terms via cosine similarity.
3. **Granger causality filtering** — Applies the Granger causality test (p < 0.05) to select lagged time series that statistically precede changes in traffic volume.
4. **Correlation filtering** — Retains only features with |Pearson correlation| > α with the target, reducing redundancy.
5. **EEMD denoising** — Ensemble Empirical Mode Decomposition removes high-frequency noise from all input signals.
6. **Causal attention mask** — Constructs a Granger causal matrix *C* applied element-wise (Hadamard product) to attention scores **before** softmax:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} \odot (1 + C)\right)V$$

where $C_{ij} = \frac{1}{\text{best\_lag}(i,j)} \cdot \mathbb{I}(X_i \to X_j)$.

### Causal Attention Layer

<p align="center">
  <img src="images/causality_layer.png" width="550"/>
</p>

*The causal attention layer modulates standard self-attention with the Granger causality matrix: scores are scaled by (1 + α·C) before normalisation.*

---

## Results

**Rural case study** — Alpujarra region, Spain. 10-fold expanding-window cross-validation (MAE ± SD, MSE ± SD, R² ± SD):

| Model | MAE ↓ | MSE ↓ | R² ↑ |
|---|---|---|---|
| ARIMA | 0.0982 ± 0.0143 | 0.0155 ± 0.0036 | −0.155 ± 0.234 |
| Univariate LSTM | 0.0684 ± 0.0134 | 0.0084 ± 0.0036 | 0.392 ± 0.233 |
| LSTM with Attention | 0.0627 ± 0.0217 | 0.0073 ± 0.0051 | 0.460 ± 0.342 |
| Causalformer | 0.0783 ± 0.0084 | 0.0101 ± 0.0018 | 0.289 ± 0.130 |
| **GCT (ours)** | **0.0438 ± 0.0143** | **0.0032 ± 0.0019** | **0.773 ± 0.132** |

### R² Score Distribution across CV folds

<p align="center">
  <img src="images/results_r2.png" width="500"/>
</p>

### Ablation Study

<p align="center">
  <img src="images/ablation_results.png" width="600"/>
</p>

| Configuration | MAE ↓ | MSE ↓ | R² ↑ |
|---|---|---|---|
| Full model | **0.0438** | **0.0032** | **0.773** |
| Without Causality filter | 0.0811 | 0.0107 | 0.206 |
| Without Correlation filter | 0.0794 | 0.0097 | 0.309 |
| Without Attention Mask | 0.0601 | 0.0060 | 0.560 |
| Base model (none) | 0.1012 | 0.0164 | −0.280 |

---

## Model Architecture (Table 1 of paper)

| Layer | Type | Shape | Key Parameters |
|---|---|---|---|
| 0 | Input | (N_batch, num\_variables) | N_batch = 64 |
| 1 | LSTM | (N_batch, 32) | lstm\_units = 32 |
| 2 | LayerNormalization | (N_batch, 32) | — |
| 3 | Multi-Head Attention + Causal Mask | (N_batch, num\_variables) | num\_heads = 4 |
| 4 | LayerNormalization | (N_batch, num\_variables) | — |
| 5 | Add (residual) | (N_batch, num\_variables) | — |
| 6 | Concatenate | (N_batch, 32 + num\_variables) | — |
| 7 | Dense (ReLU) | (N_batch, 64) | dense\_units = 64 |
| 8 | Dropout | (N_batch, 64) | dropout = 0.15 |
| 9 | Dense (ReLU) | (N_batch, 32) | — |
| 10 | Dense | (N_batch, 1) | scalar forecast |

**Training:** Adam, lr = 1×10⁻⁴, batch_size = 64, epochs = 200, 10-fold expanding-window CV.

---

## Project Structure

```
├── configs/
│   └── config.yaml          # All hyperparameters
├── images/                  # Architecture diagrams and result figures
├── scripts/
│   └── run_experiment.sh    # Full pipeline runner
└── src/
    ├── model.py             # GCT PyTorch model (canonical implementation)
    ├── preprocessing.py     # EEMD, MinMax normalisation, data loading
    ├── granger_analysis.py  # Granger tests, causal matrix, legacy Keras model
    ├── training.py          # 10-fold CV training loop, metrics, PFI
    ├── ablation.py          # 8 ablation configurations
    ├── sensitivity.py       # Sensitivity analysis (τ and α thresholds)
    └── __init__.py
```

---

## Data

Traffic counts from IoT ANPR cameras and Google Trends indices from the **Alpujarra region** (Granada, Spain). **Raw data is not included** — contact the authors for access.

---

## Quick Start

```bash
pip install -r requirements.txt

# Full GCT experiment (requires data files in ./data/)
python -m src.model \
    --trends_file data/google_trends_data.csv \
    --plates_file data/unique_num_plate_count_per_week.csv

# Ablation study
python -m src.ablation \
    --trends_file data/google_trends_data.csv \
    --plates_file data/unique_num_plate_count_per_week.csv
```

---

## Citation

If you use this code in your research, you **must** cite the following paper (CC BY 4.0 attribution requirement):

```bibtex
@article{duran2026gct,
  title={GCT: a Granger-causal transformer for multivariate traffic analysis in smart villages},
  author={Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Martinez, Daniel and De, Suparna and Bermudez-Edo, Maria},
  journal={ACM Transactions on Intelligent Systems and Technology},
  volume={17},
  number={2},
  pages={1--25},
  year={2026},
  publisher={ACM New York, NY}
}
```

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
Copyright © 2025 SmartPoqueira.  
You are free to use, adapt, and distribute this work for any purpose, including commercially, **provided you give appropriate credit and cite the paper above**.
