import numpy as np
import pandas as pd
import random
import json
import os
import matplotlib.pyplot as plt
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.sandbox.stats.runs import runstest_1samp

# Install PyEMD before running: pip install PyEMD
from PyEMD import EEMD

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation_helper(name):
    """Return an activation function by name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


def normalize_data(data):
    """Normalise data using MinMaxScaler."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, index=data.index, columns=data.columns)


def apply_eemd_denoising(data, num_imfs=5, noise_width=0.05):
    """
    Apply EEMD to each time series and reconstruct the signal using low-frequency IMFs.

    Args:
        data (pd.DataFrame): Normalised data.
        num_imfs (int): Number of IMFs to retain for reconstruction.
        noise_width (float): Noise width for EEMD.

    Returns:
        pd.DataFrame: Denoised data.
    """
    eemd = EEMD(trials=50, noise_width=noise_width)
    denoised_data = pd.DataFrame(index=data.index)

    for column in data.columns:
        signal = data[column].values
        IMFs = eemd.eemd(signal)
        # Retain the first 'num_imfs' IMFs (low frequency)
        reconstructed = np.sum(IMFs[:num_imfs], axis=0)
        denoised_data[column] = reconstructed

    return denoised_data


def load_and_prepare_data(trends_file, plates_file, normalize=True, apply_eemd=True, num_imfs=5, noise_width=0.05):
    """Load and prepare data by merging two CSV files and applying EEMD denoising."""
    trends_data = pd.read_csv(trends_file, parse_dates=['Semana'])
    plates_data = pd.read_csv(plates_file, parse_dates=['week'])

    data = pd.merge(trends_data, plates_data, left_on='Semana', right_on='week')
    data.set_index('Semana', inplace=True)
    data.drop(columns=['week'], inplace=True)

    if normalize:
        data = normalize_data(data)

    if apply_eemd:
        print("Applying Ensemble Empirical Mode Decomposition (EEMD) for denoising...")
        data = apply_eemd_denoising(data, num_imfs=num_imfs, noise_width=noise_width)
        print("EEMD denoising completed.")

    return data


def run_test(series):
    """Run a runs test on the time series."""
    _, p_value = runstest_1samp(series)
    return p_value


def granger_causality_test(data, max_lag=6):
    """
    Run Granger causality tests for each column against the target variable.
    """
    results, lags, valid_lags, run_test_results = {}, {}, {}, {}
    for col in data.columns:
        if col != 'unique_num_plate_count':
            test_result = grangercausalitytests(data[['unique_num_plate_count', col]], max_lag, verbose=False)
            p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            min_p_value = min(p_values)
            best_lag = p_values.index(min_p_value) + 1

            results[col] = min_p_value
            lags[col] = best_lag
            valid_lags[col] = [lag for lag, p_value in enumerate(p_values, 1) if p_value < 0.05]
            run_test_results[col] = run_test(data[col])

    return results, lags, valid_lags, run_test_results


def generate_shifted_trends(data, valid_lags):
    """
    Generate lagged versions of time series based on valid Granger lags.
    ""
    shifted_trends = pd.DataFrame(index=data.index)
    for trend, lag_list in valid_lags.items():
        for lag in lag_list:
            shifted_trends[f'{trend}_lag_{lag}'] = data[trend].shift(lag)
    return pd.concat([data, shifted_trends], axis=1).dropna()


def plot_loss(history, model_type="clstm", folder='plots'):
    """
    Generate and save a training and validation loss curve.
    """
    plt.figure(figsize=(10, 6))
    
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
    elif 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
    elif 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title(f'Train and Validation Loss per Epoch - {model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if 'train_loss' in history and len(history['train_loss']) > 0:
        plt.ylim([0, max(history['train_loss']) * 1.1])
    elif 'val_loss' in history and len(history['val_loss']) > 0:
        plt.ylim([0, max(history['val_loss']) * 1.1])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/training_history_{model_type}.png", dpi=500)
    plt.close()

    loss_data = {
        model_type: {
            "train_loss": history['train_loss'] if 'train_loss' in history else [],
            "val_loss": history['val_loss'] if 'val_loss' in history else []
        }
    }

    if os.path.exists(f"{folder}/loss_history.json"):
        with open(f"{folder}/loss_history.json", 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(loss_data)

    with open(f"{folder}/loss_history.json", 'w') as f:
        json.dump(existing_data, f, indent=4)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        '''
        Positional Encoding as described in "Attention is All You Need".
        
        Args:
            d_model (int): Dimension of the model.
            max_len (int): Maximum length of input sequences.
        '''
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: Tensor de shape (batch, w, d_model)
        
        Returns:
            x + pe: Tensor de shape (batch, w, d_model)
        '''
        x = x + self.pe[:, :x.size(1), :]
        return x


class TST(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, output_dim=1, context=6):
        '''
        Time Series Transformer (TST) model.
        
        Args:
            input_dim (int): Number of input variables (m).
            model_dim (int): Dimension of the Transformer model (d).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            output_dim (int): Dimension of the output (n).
            context (int): Size of the input window.
        '''
        super(TST, self).__init__()
        self.model_dim = model_dim
        self.context = context  # Guardar el contexto
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True  # Para evitar el warning sobre batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.output_linear = nn.Linear(model_dim * context, output_dim)  # Ajustado para context=6

    def forward(self, x):
        '''
        Args:
            x: Tensor de shape (batch, w, m)
        
        Returns:
            y_pred: Tensor de shape (batch, n)
        '''
        # Linear transformation
        x = self.input_linear(x)  # (batch, w, d)
        # Add positional encoding
        x = self.positional_encoding(x)  # (batch, w, d)
        # Transformer expects input of shape (batch, w, d) with batch_first=True
        x = self.transformer_encoder(x)  # (batch, w, d)
        # Flatten
        x = self.flatten(x)  # (batch, w*d)
        # Output layer
        y_pred = self.output_linear(x)  # (batch, n)
        return y_pred


class CGTST(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, output_dim=1, context=6):
        '''
        Causality-Gated Time Series Transformer (CGTST) model.
        
        Args:
            input_dim (int): Number of input variables (m).
            model_dim (int): Dimension of the Transformer model (d).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            output_dim (int): Dimension of the output (n).
            context (int): Size of the input window.
        '''
        super(CGTST, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.context = context  

        self.causality_gate = nn.Parameter(torch.ones(input_dim))

        self.tst = TST(input_dim, model_dim, num_heads, num_layers, dropout, output_dim, context)

    def forward(self, x):
        '''
        Args:
            x: Tensor de forma (batch, w, m)
        
        Returns:
            y_pred: Tensor de forma (batch, n)
        '''
        # Apply softmax to causality gates
        gate = torch.softmax(self.causality_gate, dim=0)  # (m,)
        # Expand gate to match x's dimensions
        gate = gate.unsqueeze(0).unsqueeze(1)  # (1, 1, m)
        # Multiply input variables by gate
        x = x * gate  # (batch, w, m)
        # Pass through TST
        y_pred = self.tst(x)  # (batch, n)
        return y_pred

    def get_causality_gates(self):
        '''
        Returns the causality gates after softmax.
        '''
        return torch.softmax(self.causality_gate, dim=0).detach().cpu().numpy()


def regularize(network, lam):
    '''
    Calculate regularization term for causality gates and other parameters.
    
    Args:
      network: CGTST network.
      lam: regularization parameter.
    '''
    # Regularize causality gates (encourage sparsity)
    return lam * torch.sum(torch.abs(network.causality_gate))


def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layers and hidden-hidden weights.'''
    return lam * (
        torch.sum(network.tst.input_linear.weight ** 2) +
        torch.sum(network.tst.output_linear.weight ** 2) +
        torch.sum(network.tst.transformer_encoder.layers[0].linear1.weight ** 2) +
        torch.sum(network.tst.transformer_encoder.layers[0].linear2.weight ** 2)
    )


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params.data.clone()


def arrange_input(data, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    if data.ndimension() == 2:
        T, dim = data.shape
    else:
        raise ValueError("Data must be a 2D tensor of shape (T, dim).")
    input_seq = torch.zeros(T - context, context, dim,
                            dtype=torch.float32, device=data.device)
    target_seq = torch.zeros(T - context, dim,
                             dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = T - context + i
        input_seq[:, i, :] = data[start:end]
    target_seq = data[context:T]
    return input_seq.detach(), target_seq.detach()


def train_model_cgtst(cgtst, X_train, Y_train, X_val, Y_val, lr, max_iter, lam=0, lam_ridge=0,
                     check_every=1, verbose=1):
    '''Train CGTST model with Adam optimizer.'''
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cgtst.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}

    for it in range(max_iter):
        # Forward pass on training data.
        cgtst.train()
        y_pred_train = cgtst(X_train)  # (batch, n)
        loss = loss_fn(y_pred_train, Y_train)  # Assuming Y_train is (batch, n)

        # Add regularization terms
        reg = 0.0
        if lam > 0:
            reg += regularize(cgtst, lam)
        if lam_ridge > 0:
            reg += ridge_regularize(cgtst, lam_ridge)
        loss = loss + reg

        # Backward pass and optimization step.
        loss.backward()
        optimizer.step()
        cgtst.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            cgtst.eval()
            with torch.no_grad():
                y_pred_val = cgtst(X_val)  # (batch, n)
                val_loss = loss_fn(y_pred_val, Y_val)
                history['train_loss'].append(loss.item())
                history['val_loss'].append(val_loss.item())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print(f'Train Loss = {loss.item():.6f}')
                print(f'Validation Loss = {val_loss.item():.6f}')

    return history


def permutation_feature_importance(model, X_val, Y_val, selected_columns, threshold=1.0):
    """
    Permutation Feature Importance (PFI) to validate causal relationships.

    Args:
        model (CGTST): Trained model.
        X_val (torch.Tensor): Validation input data.
        Y_val (torch.Tensor): Validation target data.
        selected_columns (list): Feature column names.
        threshold (float): PFI ratio threshold below which a variable is considered causal.

    Returns:
        dict: Validated causal relationships.
    """
    model.eval()
    loss_fn = nn.MSELoss(reduction='mean')
    original_loss = loss_fn(model(X_val), Y_val).item()
    print(f"Original Validation Loss: {original_loss:.6f}")

    causality_gates = model.get_causality_gates()
    causal_relationships = {}

    for i, col in enumerate(selected_columns):
        # Permute the feature along the temporal dimension
        X_permuted = X_val.clone()
        perm_indices = torch.randperm(X_permuted.size(1))
        X_permuted[:, :, i] = X_permuted[:, perm_indices, i]
        
        # Calcular la pérdida con la columna permutada
        with torch.no_grad():
            y_pred_permuted = model(X_permuted)
            permuted_loss = loss_fn(y_pred_permuted, Y_val).item()
        
        # PFI ratio: < threshold means the variable is causal
        pfi_ratio = original_loss / permuted_loss if permuted_loss != 0 else float('inf')
        print(f"PFI Ratio for {col}: {pfi_ratio:.6f}")

        if pfi_ratio < threshold:
            causal_relationships[col] = pfi_ratio

    return causal_relationships


def prepare_data_for_model_cv(
    data, 
    selected_columns, 
    clstm_params, 
    training_params, 
    n_splits=5, 
    min_train_ratio=0.5, 
    max_train_ratio=0.8, 
    test_ratio=0.2, 
    case='clstm'
):
    """
    Prepara los datos y entrena el modelo usando validación cruzada.
    """
    X = data[selected_columns].values  # Input features
    y = data['unique_num_plate_count'].values  # Target variable

    total_samples = len(X)
    test_size = int(test_ratio * total_samples)
    min_train_size = int(min_train_ratio * total_samples)
    max_train_size = int(max_train_ratio * total_samples)
    if n_splits > 1:
        train_size_increase = (max_train_size - min_train_size) // (n_splits - 1)
    else:
        train_size_increase = 0

    splits = []

    for fold in range(n_splits):
        train_size = min_train_size + fold * train_size_increase
        start_test = train_size
        end_test = start_test + test_size

        start_train = 0
        end_train = start_test

        # Asegurar que end_test no exceda el total de muestras
        if end_test > total_samples:
            end_test = total_samples

        splits.append((np.arange(start_train, end_train), np.arange(start_test, end_test)))

    fold_summaries = []
    histories = []
    models = []
    validation_data = []  # List to store validation data for each fold

    total_time = 0        # Accumulator for total execution time

    os.makedirs(FOLDER, exist_ok=True)

    with open(f"{FOLDER}/cgtst_pytorch.txt", 'w') as f:
        f.write(f"\n\n--- {case.capitalize()} Model Results ---\n")
        f.write(f"Total number of instances: {total_samples}\n")

        for fold, (train_index, test_index) in enumerate(splits):
            # Verificar que el tamaño de entrenamiento es suficiente
            context = clstm_params.get('context', 6)
            min_required_train = context + 1  # Debe ser al menos context + 1
            if len(train_index) < min_required_train:
                print(f"Fold {fold + 1} skipped: insufficient training samples ({len(train_index)} < {min_required_train})")
                f.write(f"\nFold {fold + 1} skipped: insufficient training samples ({len(train_index)} < {min_required_train})\n")
                continue

            start_time_fold = time.time()

            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            # Ajustar la forma de los datos
            context = clstm_params.get('context', 6)
            hidden = clstm_params.get('hidden', 64)
            batch_size = training_params.get('batch_size', 128)

            num_input_series = X_train.shape[1]  # Número de características de entrada

            # Convertir a tensores
            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            Y_train_tensor = torch.from_numpy(y_train).float().to(device)  # (batch, )
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            Y_val_tensor = torch.from_numpy(y_val).float().to(device)    # (batch, )

            # Arrange input
            try:
                X_train_arranged, Y_train_arranged = arrange_input(X_train_tensor, context)
                X_val_arranged, Y_val_arranged = arrange_input(X_val_tensor, context)
            except ValueError as e:
                print(f"Fold {fold + 1} skipped: {e}")
                f.write(f"\nFold {fold + 1} skipped: {e}\n")
                continue

            # Inicializar el modelo CGTST con el contexto y output_dim correcto
            output_dim = Y_train_arranged.shape[1] if Y_train_arranged.ndimension() > 1 else 1
            model = CGTST(
                input_dim=num_input_series,
                model_dim=clstm_params.get('model_dim', 64),
                num_heads=clstm_params.get('num_heads', 4),
                num_layers=clstm_params.get('num_layers', 2),
                dropout=clstm_params.get('dropout', 0.1),
                output_dim=output_dim,  # Predicción de un timestep
                context=context  # Pasar el contexto
            ).to(device)

            # Definir hiperparámetros de entrenamiento
            optimizer_type = training_params.get('optimizer', 'adam')  # Solo 'adam'
            learning_rate = training_params.get('learning_rate', 0.01)
            epochs = training_params.get('epochs', 200)
            lam = training_params.get('lam', 0.0)
            lam_ridge = training_params.get('lam_ridge', 0.0)

            # Entrenar el modelo utilizando Adam sin early stopping
            if optimizer_type == 'adam':
                history = train_model_cgtst(
                    model, X_train_arranged, Y_train_arranged,
                    X_val_arranged, Y_val_arranged,
                    learning_rate, epochs,
                    lam=lam, lam_ridge=lam_ridge,
                    check_every=training_params.get('check_every', 1),
                    verbose=training_params.get('verbose', 1)
                )
            else:
                raise ValueError("Unsupported optimizer type. Only 'adam' is supported.")

            histories.append(history)
            models.append(model)
            validation_data.append((X_val_arranged, Y_val_arranged))  # Almacenar datos de validación

            # Predecir en el conjunto de validación
            model.eval()
            with torch.no_grad():
                predictions = model(X_val_arranged)
                predictions = predictions.cpu().numpy()  # (batch, n)
                y_val_seq = Y_val_arranged.cpu().numpy()  # (batch, n)

            # Verificar las longitudes antes de calcular métricas
            print(f"Fold {fold + 1}: y_val_seq shape = {y_val_seq.shape}, predictions shape = {predictions.shape}")
            f.write(f"\nFold {fold + 1}: y_val_seq shape = {y_val_seq.shape}, predictions shape = {predictions.shape}\n")

            if y_val_seq.shape != predictions.shape:
                print(f"Fold {fold + 1} skipped: y_val_seq shape {y_val_seq.shape} != predictions shape {predictions.shape}")
                f.write(f"\nFold {fold + 1} skipped: y_val_seq shape {y_val_seq.shape} != predictions shape {predictions.shape}\n")
                continue

            # Calcular métricas
            mae = mean_absolute_error(y_val_seq, predictions)
            mse = mean_squared_error(y_val_seq, predictions)
            r2 = r2_score(y_val_seq, predictions)

            fold_summary = {
                'fold': fold + 1,
                'train_size': len(train_index),
                'test_size': len(test_index),
                'train_range': (train_index[0], train_index[-1]),
                'test_range': (test_index[0], test_index[-1]),
                'mae': mae,
                'mse': mse,
                'r2': r2
            }
            fold_summaries.append(fold_summary)

            f.write(f"\nFold {fold_summary['fold']} Summary:\n")
            f.write(f"Training samples: {fold_summary['train_size']}, Testing samples: {fold_summary['test_size']}\n")
            f.write(f"Training range: {fold_summary['train_range']}\n")
            f.write(f"Testing range: {fold_summary['test_range']}\n")
            f.write(f"MAE: {fold_summary['mae']:.4f}\n")
            f.write(f"MSE: {fold_summary['mse']:.4f}\n")
            f.write(f"R2: {fold_summary['r2']:.4f}\n")

            # Calcular tiempo de ejecución para el fold
            fold_execution_time = time.time() - start_time_fold
            total_time += fold_execution_time
            f.write(f"Execution Time (Fold {fold + 1}): {fold_execution_time:.2f} seconds\n")

            # Almacenar los causality gates para cada fold
            gates = model.get_causality_gates()
            f.write(f"Causality Gates (Fold {fold + 1}): {gates.tolist()}\n")

        # Cálculos finales
        if fold_summaries:
            avg_mae = np.mean([s['mae'] for s in fold_summaries])
            avg_mse = np.mean([s['mse'] for s in fold_summaries])
            avg_r2 = np.mean([s['r2'] for s in fold_summaries])
            avg_execution_time = total_time / len(fold_summaries)  # Tiempo promedio de ejecución

            f.write(f"\n{case.capitalize()} case - Average MAE: {avg_mae:.4f}\n")
            f.write(f"{case.capitalize()} case - Average MSE: {avg_mse:.4f}\n")
            f.write(f"{case.capitalize()} case - Average R2: {avg_r2:.4f}\n")
            f.write(f"{case.capitalize()} case - Average Execution Time: {avg_execution_time:.2f} seconds\n")
        else:
            f.write(f"\n{case.capitalize()} case - No valid folds found.\n")

        # Graficar la última historia de entrenamiento
        if histories:
            last_history = histories[-1]
            # Para train_model_cgtst, history es un dict con 'train_loss' y 'val_loss'
            plot_loss(last_history, model_type="cgtst_pytorch", folder=FOLDER)
        else:
            print("No valid training history found. Please check the data splits and parameters.")

    return last_history, models, fold_summaries, validation_data


def main():
    parser = argparse.ArgumentParser(description="CGTST - Causality-Gated Time Series Transformer Model")

    # Argumentos de semilla y carpeta
    parser.add_argument('--seed', type=int, default=161, help='Semilla para reproducibilidad')
    parser.add_argument('--folder', type=str, default='plots161', help='Carpeta para guardar los gráficos y registros')

    # Argumentos de rutas de archivos
    parser.add_argument('--trends_file', type=str, default='./DB/google_trends_data.csv', help='Ruta del archivo de tendencias de Google')
    parser.add_argument('--plates_file', type=str, default='./DB/unique_num_plate_count_per_week.csv', help='Ruta del archivo de conteo de placas únicas por semana')

    # Argumentos de hiperparámetros del modelo (ajustados para mejorar rendimiento)
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Tasa de aprendizaje del optimizador')
    parser.add_argument('--epochs', type=int, default=200, help='Número de épocas de entrenamiento')
    parser.add_argument('--input_window', type=int, default=6, help='Tamaño de la ventana de entrada temporal')

    # Argumentos adicionales para regularización
    parser.add_argument('--lam', type=float, default=1e-3, help='Coeficiente de regularización lambda_K')
    parser.add_argument('--lam_ridge', type=float, default=1e-3, help='Coeficiente de regularización lambda_M')

    # Argumentos de validación cruzada
    parser.add_argument('--n_splits', type=int, default=10, help='Número de splits de validación cruzada')
    parser.add_argument('--min_train_ratio', type=float, default=0.5, help='Proporción mínima de datos de entrenamiento')
    parser.add_argument('--max_train_ratio', type=float, default=0.8, help='Proporción máxima de datos de entrenamiento')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Proporción de datos de prueba')

    # Nuevo argumento para batch size
    parser.add_argument('--batch_size', type=int, default=128, help='Tamaño del lote para DataLoaders')

    # Argumentos específicos para CGTST
    parser.add_argument('--hidden', type=int, default=64, help='Número de unidades ocultas para CGTST')
    parser.add_argument('--activation', type=str, default='relu', help='Función de activación para CGTST')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'], help='Optimizador a utilizar: "adam"')
    parser.add_argument('--penalty', type=str, default='H', choices=['GL', 'GSGL', 'H'], help='Tipo de regularización: "GL", "GSGL", "H"')
    parser.add_argument('--lookback', type=int, default=5, help='Número de checks antes de early stopping para adam (no utilizado)')
    parser.add_argument('--check_every', type=int, default=1, help='Frecuencia para registrar la pérdida')
    parser.add_argument('--verbose', type=int, default=1, help='Nivel de verbosidad (0, 1, 2)')

    args = parser.parse_args()

    # Asignar los argumentos a variables globales para acceso en otras funciones
    global SEED, FOLDER
    SEED = args.seed
    FOLDER = args.folder
    set_seed(SEED)

    # Definir rutas de archivos
    trends_file = args.trends_file
    plates_file = args.plates_file

    # Cargar y normalizar datos con EEMD
    data = load_and_prepare_data(
        trends_file, 
        plates_file, 
        normalize=True, 
        apply_eemd=True, 
        num_imfs=5, 
        noise_width=0.05
    )

    # **1. Usar Todas las Tendencias como Características**
    # Excluir la variable objetivo 'unique_num_plate_count'
    trend_columns = data.columns
    selected_columns = trend_columns.tolist()

    print(f"Using all trends for the model: {selected_columns}")


    # **3. Seleccionar Todas las Características Originales Sin Desplazamiento**
    # Las columnas de características son todas las tendencias excluyendo la variable objetivo
    feature_columns = selected_columns.copy()

    print(f"Total features without shifting: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns}")

    # **4. Definir Parámetros del Modelo CGTST**
    clstm_params = {
        'num_input_series': len(feature_columns),
        'hidden': args.hidden,
        'context': args.input_window,       # Definir 'context' para CGTST
        'model_dim': 64,                    # Dimensión del modelo Transformer
        'num_heads': 4,                     # Número de heads en atención multi-head
        'num_layers': 2,                    # Número de capas del Transformer
        'dropout': 0.1                      # Tasa de dropout
    }

    # **5. Definir Parámetros de Entrenamiento**
    training_params = {
        'optimizer': args.optimizer,       # Solo 'adam'
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'lam': args.lam,
        'lam_ridge': args.lam_ridge,
        'penalty': args.penalty,
        'lookback': args.lookback,          # No se usará para early stopping
        'check_every': args.check_every,    # Ahora 1 para registrar cada época
        'verbose': args.verbose,
        'batch_size': args.batch_size
    }

    print("Starting model training with CGTST and specified window size...")
    last_history, models, fold_summaries, validation_data = prepare_data_for_model_cv(
        data,                # Usar los datos originales sin características desplazadas
        feature_columns,     # Todas las columnas de características seleccionadas
        clstm_params,        # Parámetros del modelo
        training_params,     # Parámetros de entrenamiento
        n_splits=args.n_splits,          # Número de splits de validación cruzada
        min_train_ratio=args.min_train_ratio,
        max_train_ratio=args.max_train_ratio,
        test_ratio=args.test_ratio,
        case='cgtst'
    )

    # **6. Implementar PFI para Validar Relaciones Causales**
    print("Starting Permutation Feature Importance (PFI) for causality validation...")

    # Implementación de PFI para cada modelo entrenado en cada fold
    causal_relationships_all_folds = []
    for idx, model in enumerate(models):
        print(f"\nApplying PFI for Fold {idx + 1}")
        X_val_arranged, Y_val_arranged = validation_data[idx]
        causal_relationships = permutation_feature_importance(
            model, 
            X_val=X_val_arranged, 
            Y_val=Y_val_arranged, 
            selected_columns=feature_columns, 
            threshold=1.0
        )
        causal_relationships_all_folds.append(causal_relationships)
        print(f"Causal Relationships for Fold {idx + 1}: {causal_relationships}")

    # **7. Graficar Pérdida de Entrenamiento y Validación**
    if last_history and 'train_loss' in last_history and 'val_loss' in last_history:
        plot_loss(last_history, model_type="cgtst_pytorch", folder=FOLDER)
        print("Model training and evaluation completed successfully.")
    else:
        print("No valid training history found. Please check the data splits and parameters.")


if __name__ == "__main__":
    main()
