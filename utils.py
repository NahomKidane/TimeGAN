"""
Time-series Generative Adversarial Networks (TimeGAN) Codebase - TF2.x Version

Converted from original TF1.x version (Jinsung Yoon et al., 2019).
Maintains the same logic, but uses tf.keras & NumPy arrays for TF2 compatibility.

Functions:
(1) train_test_divide
(2) extract_time
(3) rnn_cell (returns string; now handled in main code)
(4) random_generator
(5) batch_generator
"""

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# 1. Train/Test Split
# -----------------------------------------------------------------------------
def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

# -----------------------------------------------------------------------------
# 2. Extract Time
# -----------------------------------------------------------------------------
def extract_time(data):
    """Returns time lengths and max sequence length."""
    time = [len(seq) for seq in data]
    max_seq_len = max(time)
    return time, max_seq_len

# -----------------------------------------------------------------------------
# 3. RNN Cell Placeholder (TF2 uses keras layers instead)
# -----------------------------------------------------------------------------
def rnn_cell(module_name, hidden_dim):
    """
    For compatibility with original code.
    In TF2, we use tf.keras.layers directly in timegan.py.
    """
    return module_name, hidden_dim

# -----------------------------------------------------------------------------
# 4. Random Vector Generator
# -----------------------------------------------------------------------------
def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = np.zeros((batch_size, max_seq_len, z_dim))
    for i in range(batch_size):
        temp_Z = np.random.uniform(0., 1., (T_mb[i], z_dim))
        Z_mb[i, :T_mb[i], :] = temp_Z
    return Z_mb.astype(np.float32)

# -----------------------------------------------------------------------------
# 5. Batch Generator (TF2 padded)
# -----------------------------------------------------------------------------
def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    batch_idx = idx[:batch_size]

    max_seq_len = max([time[i] for i in batch_idx])
    dim = data[0].shape[-1]

    X_mb = np.zeros((batch_size, max_seq_len, dim))
    T_mb = []

    for i, j in enumerate(batch_idx):
        seq_len = time[j]
        X_mb[i, :seq_len, :] = data[j]
        T_mb.append(seq_len)

    return X_mb.astype(np.float32), np.array(T_mb, dtype=np.int32)
