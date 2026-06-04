import numpy as np
import pandas as pd
import random
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.sandbox.stats.runs import runstest_1samp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Concatenate, Add, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import matplotlib.pyplot as plt

import sys

if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    SEED = 11 


import sys
if len(sys.argv) > 1:
    FOLDER = sys.argv[2]
else:
    FOLDER = 'plots' 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)



def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

def load_and_prepare_data(trends_file, plates_file, normalize=True):
    trends_data = pd.read_csv(trends_file, parse_dates=['Semana'])
    plates_data = pd.read_csv(plates_file, parse_dates=['week'])

    data = pd.merge(trends_data, plates_data, left_on='Semana', right_on='week')
    data.set_index('Semana', inplace=True)
    data.drop(columns=['week'], inplace=True)

    if normalize:
        data = normalize_data(data)
    return data

def run_test(series):
    _, p_value = runstest_1samp(series)
    return p_value

def granger_causality_test(data, max_lag=6):
    results, lags, valid_lags, run_test_results = {}, {}, {}, {}
    for col in data.columns:
        if col != 'unique_num_plate_count':
            test_result = grangercausalitytests(data[['unique_num_plate_count', col]], max_lag, verbose=False)
            p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            min_p_value = min(p_values)
            best_lag = p_values.index(min_p_value) + 1

            results[col] = min_p_value
            lags[col] = best_lag
            valid_lags[col] = [lag + 1 for lag, p_value in enumerate(p_values) if p_value < 0.05]
            run_test_results[col] = run_test(data[col])

    return results, lags, valid_lags, run_test_results

def generate_shifted_trends(data, valid_lags):
    shifted_trends = pd.DataFrame(index=data.index)
    for trend, lag_list in valid_lags.items():
        for lag in lag_list:
            shifted_trends[f'{trend}_lag_{lag}'] = data[trend].shift(lag)
    return pd.concat([data, shifted_trends], axis=1).dropna()


def plot_loss(history, model_type="multivariate_causality"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Train and Validation Loss per Epoch - {model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 0.5])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{FOLDER}/training_history_{model_type}.png", dpi=500)
    plt.close()

    loss_data = {
        model_type: {
            "loss": history.history['loss'],
            "val_loss": history.history['val_loss']
        }
    }

    if os.path.exists(f"{FOLDER}/loss_history.json"):
        with open(f"{FOLDER}/loss_history.json", 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {} 

    existing_data.update(loss_data)

    with open(f"{FOLDER}/loss_history.json", 'w') as f:
        json.dump(existing_data, f, indent=4)


"""
class SelfAttentionWithCausality(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, causal_matrix, alpha):
        super(SelfAttentionWithCausality, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.causal_matrix = causal_matrix
        self.alpha = alpha
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        if self.alpha != 0:
            causal_adjustment = tf.matmul(inputs, self.causal_matrix) * self.alpha
            adjusted_attention = attention_output + causal_adjustment 
            return adjusted_attention
        else:
            return attention_output
"""

class SelfAttentionWithCausality(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, causal_matrix, alpha):
        super(SelfAttentionWithCausality, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.causal_matrix = causal_matrix
        self.alpha = alpha
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)

        if self.alpha != 0:
            causal_adjustment = tf.matmul(inputs, self.causal_matrix)

            causal_adjustment = 1 + self.alpha * causal_adjustment

                    # Hadamard product with causal adjustment
            adjusted_attention = attention_output * causal_adjustment
            return adjusted_attention
        else:
            return attention_output





def create_causal_matrix(data, selected_trends, lags, max_lag=6):
    num_trends = len(selected_trends)
    causal_matrix = np.zeros((num_trends, num_trends))

    for i, trend_i in enumerate(selected_trends):
        for j, trend_j in enumerate(selected_trends):
            if i != j:
                # Run Granger causality test to determine causal direction
                result = grangercausalitytests(data[[trend_j, trend_i]], max_lag, verbose=False)
                p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                min_p_value = min(p_values)
                if min_p_value < 0.05:  # Only consider relationships with significant p-value
                    best_lag = p_values.index(min_p_value) + 1
                    causal_matrix[i, j] = 1/best_lag  # Inverse of best lag as causality strength
                else:
                    causal_matrix[i, j] = 0  # No significant causality

    return causal_matrix

def create_transformer_model(input_shape, params, causal_matrix, alpha):
    lstm_units = params['lstm_units']
    #num_heads = params['num_heads']
    #key_dim = params['key_dim']
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    dense_units = params['dense_units']

    num_trends = causal_matrix.shape[0]

    inputs = Input(shape=input_shape)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = LayerNormalization(epsilon=1e-6)(lstm_out)

    # Dense Layer for projection before self-attention
    dense_proj = Dense(num_trends, use_bias=False)(lstm_out)

    # Self-Attention with Causality Layer using projected input
    attention = SelfAttentionWithCausality(num_heads=4, key_dim=num_trends, causal_matrix=causal_matrix, alpha=alpha)(dense_proj)
    attention = LayerNormalization(epsilon=1e-6)(attention)

    add = Add()([dense_proj, attention])
    concat = Concatenate()([inputs, add])

    dense_out = Dense(dense_units, activation='relu')(concat)
    dense_out = Dropout(dropout_rate)(dense_out)
    dense_out = Dense(dense_units//2, activation='relu')(dense_out)
    dense_out = Dropout(dropout_rate)(dense_out)
    outputs = Dense(1)(dense_out)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model






import numpy as np
import time  # measure fold execution time

def prepare_data_for_model_cv(data, selected_columns, params, causal_matrix, alpha, n_splits=5, min_train_ratio=0.5, max_train_ratio=0.8, test_ratio=0.2, case='multivariate_causality'):
    X = data[selected_columns + ['unique_num_plate_count']].values
    y = data['unique_num_plate_count'].values

    total_samples = len(X)
    test_size = int(test_ratio * total_samples)
    min_train_size = int(min_train_ratio * total_samples)
    max_train_size = int(max_train_ratio * total_samples)
    train_size_increase = (max_train_size - min_train_size) // (n_splits - 1)

    splits = []

    for fold in range(n_splits):
        train_size = min_train_size + fold * train_size_increase
        start_test = train_size
        end_test = start_test + test_size

        start_train = 0
        end_train = start_test

        splits.append((np.arange(start_train, end_train), np.arange(start_test, end_test)))

    fold_summaries = []
    histories = []

    total_time = 0  # accumulates total execution time across folds

    with open(f"{FOLDER}/multivariate_causality.txt", 'w') as f:
        f.write(f"\n\n--- {case.capitalize()} Model Results ---\n")
        f.write(f"Total number of instances: {total_samples}\n")

        for fold, (train_index, test_index) in enumerate(splits):
            start_time = time.time()  # start fold timer

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            model = create_transformer_model((X_train.shape[1], X_train.shape[2]), params, causal_matrix, alpha)
            
            try:
                history = model.fit(X_train, y_train.reshape(-1, 1), epochs=params['epochs'], validation_data=(X_test, y_test.reshape(-1, 1)), batch_size=16)
                histories.append(history)  # Store the training history
            except Exception as e:
                f.write(f'Error during model.fit: {e}\n')
                continue


            predictions = model.predict(X_test).reshape(-1)
            y_test = y_test.reshape(-1)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

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

            # Compute fold execution time
            fold_execution_time = time.time() - start_time
            total_time += fold_execution_time
            f.write(f"Execution Time (Fold {fold + 1}): {fold_execution_time:.2f} seconds\n")

        # Final aggregated metrics
        if fold_summaries:
            avg_mae = np.mean([s['mae'] for s in fold_summaries])
            avg_mse = np.mean([s['mse'] for s in fold_summaries])
            avg_r2 = np.mean([s['r2'] for s in fold_summaries])
            avg_execution_time = total_time / n_splits  # average execution time per fold

            f.write(f"\n{case.capitalize()} case - Average MAE: {avg_mae:.4f}\n")
            f.write(f"{case.capitalize()} case - Average MSE: {avg_mse:.4f}\n")
            f.write(f"{case.capitalize()} case - Average R2: {avg_r2:.4f}\n")
            f.write(f"{case.capitalize()} case - Average Execution Time: {avg_execution_time:.2f} seconds\n")
        else:
            f.write(f"\n{case.capitalize()} case - No valid folds found.\n")

    return histories[-1]



def main():
    trends_file = './DB/google_trends_data.csv'
    plates_file = './DB/unique_num_plate_count_per_week.csv'

    data = load_and_prepare_data(trends_file, plates_file, normalize=True)

    results, lags, valid_lags, run_test_results = granger_causality_test(data)
    
    # Select trends that pass the Granger test at p < 0.05
    selected_trends = [trend for trend, p_value in results.items() if p_value < 0.05]

    print(f"Selected trends for forecasting: {selected_trends}")

    # Generate lagged versions based on valid Granger lags
    all_trends = generate_shifted_trends(data, valid_lags)

    # Select columns highly correlated (>= 0.5) with the target and present in selected_trends
    correlations = all_trends.corr()['unique_num_plate_count']
    selected_columns = [col for col in all_trends.columns if correlations[col] >= 0.5 and any(trend in col for trend in selected_trends)]

    print(f"Feature columns selected for the multivariate model: {selected_columns}")


    alpha = 1
    causal_matrix = create_causal_matrix(data, selected_trends, lags)

    print(causal_matrix)

    transformer_params = {
        'lstm_units': 32,
        'num_heads': 4,
        'key_dim': 32,
        'dropout_rate': 0.15,
        'learning_rate': 0.0001,
        'dense_units': 64,
        'epochs': 200
    }

    print("Multivariate Model (Selected Trends):")
    last_history = prepare_data_for_model_cv(all_trends, selected_columns, transformer_params, causal_matrix, alpha, n_splits=10, case='multivariate_causality')

    plot_loss(last_history)


if __name__ == "__main__":
    main()
