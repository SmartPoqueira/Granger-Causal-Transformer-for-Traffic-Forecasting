# GCT: Granger-Causal Transformer for Traffic Flow Forecasting

A time-series forecasting model that integrates **Granger causality analysis** with a **Transformer architecture** to predict traffic flow in smart villages. The model uses Google Trends data as exogenous signals, applying EEMD denoising and a learned causality gate to weight feature contributions based on their causal relationship with the target variable.

## Overview

Predicting traffic flow in rural areas is challenging due to limited sensor infrastructure and seasonal variability. GCT addresses this by:

1. **Fusing heterogeneous data** — Combining IoT traffic counts with Google Trends search interest data.
2. **Granger causality filtering** — Identifying which external signals causally precede changes in traffic volume.
3. **Causal attention biasing** — Constructing a causality matrix $C_{ij} = \frac{1}{\text{best\_lag}} \cdot \mathbb{I}(X_i \to X_j)$ that modulates transformer attention weights via Hadamard product.
4. **EEMD denoising** — Ensemble Empirical Mode Decomposition removes high-frequency noise from all input signals before modeling.

## Method

The CGTST (Causality-Gated Time Series Transformer) model:

- **Input**: Multivariate time series of shape `(T, m)` where `m` is the number of input features.
- **Causality Gate**: Learnable softmax-gated vector that weights each input variable.
- **Architecture**: Linear projection → Positional Encoding → Transformer Encoder → Flatten → Output Linear.
- **Causal Attention**: Standard multi-head attention is modulated by the Granger causality matrix using `attention * (1 + α·C)`.

## Results

Expanding-window cross-validation (10 folds) on the Alpujarra smart village dataset:

| Model | MAE ↓ | MSE ↓ | R² ↑ |
|---|---|---|---|
| LSTM | 0.0847 | 0.0123 | 0.721 |
| Transformer (TST) | 0.0792 | 0.0108 | 0.755 |
| CGTST (no causality) | 0.0768 | 0.0101 | 0.771 |
| **CGTST (Granger)** | **0.0634** | **0.0072** | **0.836** |
| CGTST (Granger + Correlation) | 0.0651 | 0.0076 | 0.828 |

## Project Structure

```
GCT/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── model.py              # CGTST model (TST + causality gate)
│   ├── granger_analysis.py   # Granger tests + causal matrix
│   ├── data_loader.py        # Data loading + EEMD denoising
│   ├── train.py              # Expanding-window CV training
│   └── utils.py
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
└── scripts/
    └── run_experiment.sh
```

## Data

Traffic count data and Google Trends indices from the Alpujarra region (Granada, Spain). Data is not included in this repository.

- **Traffic counts**: Weekly unique license plate counts from IoT cameras.
- **Google Trends**: Search interest for tourism-related queries (accommodation, hiking, rural tourism, etc.).

## Quick Start

```bash
pip install -r requirements.txt
python -m src.train --config configs/config.yaml
```

## Citation

```bibtex
@inproceedings{duranlopez2025gct,
  title={Granger-Causal Transformer for Traffic Flow Forecasting in Smart Villages},
  author={Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Mart{\'i}nez, Daniel and Berm{\'u}dez-Edo, Mar{\'i}a and De, Suparna},
  booktitle={Proceedings of the ACM Conference},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE).
