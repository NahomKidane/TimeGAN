"""
Time-series Generative Adversarial Networks (TimeGAN) - TensorFlow 2.x Version

Converted from original TF1.x by Jinsung Yoon et al., 2019 (NeurIPS paper)

Maintained same logic & training steps, but adapted to eager execution and tf.keras.
"""

import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator

# -----------------------------------------------------------------------------
# Min-Max Normalization (unchanged)
# -----------------------------------------------------------------------------
def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

# -----------------------------------------------------------------------------
# Model Components: Embedder, Recovery, Generator, Supervisor, Discriminator
# -----------------------------------------------------------------------------
class RNNBlock(tf.keras.layers.Layer):
    def __init__(self, module_name, hidden_dim, return_sequences=True):
        super().__init__()
        if module_name.lower() == 'lstm':
            self.rnn = tf.keras.layers.LSTM(hidden_dim, return_sequences=return_sequences)
        else:
            self.rnn = tf.keras.layers.GRU(hidden_dim, return_sequences=return_sequences)

    def call(self, x):
        return self.rnn(x)

class Embedder(tf.keras.Model):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnns = [RNNBlock(module_name, hidden_dim) for _ in range(num_layers)]
        self.fc = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, x):
        h = x
        for r in self.rnns:
            h = r(h)
        return self.fc(h)

class Recovery(tf.keras.Model):
    def __init__(self, module_name, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnns = [RNNBlock(module_name, hidden_dim) for _ in range(num_layers)]
        self.fc = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, h):
        out = h
        for r in self.rnns:
            out = r(out)
        return self.fc(out)

class Generator(tf.keras.Model):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnns = [RNNBlock(module_name, hidden_dim) for _ in range(num_layers)]
        self.fc = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, z):
        out = z
        for r in self.rnns:
            out = r(out)
        return self.fc(out)

class Supervisor(tf.keras.Model):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnns = [RNNBlock(module_name, hidden_dim) for _ in range(num_layers - 1)]
        self.fc = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

    def call(self, h):
        out = h
        for r in self.rnns:
            out = r(out)
        return self.fc(out)

class Discriminator(tf.keras.Model):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        self.rnns = [RNNBlock(module_name, hidden_dim) for _ in range(num_layers)]
        self.fc = tf.keras.layers.Dense(1, activation=None)

    def call(self, h):
        out = h
        for r in self.rnns:
            out = r(out)
        return self.fc(out)

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss(y_real, y_fake, y_fake_e, gamma):
    d_loss_real = bce(tf.ones_like(y_real), y_real)
    d_loss_fake = bce(tf.zeros_like(y_fake), y_fake)
    d_loss_fake_e = bce(tf.zeros_like(y_fake_e), y_fake_e)
    return d_loss_real + d_loss_fake + gamma * d_loss_fake_e

def generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat, gamma):
    # Adversarial
    g_loss_u = bce(tf.ones_like(y_fake), y_fake)
    g_loss_u_e = bce(tf.ones_like(y_fake_e), y_fake_e)

    # Supervised
    g_loss_s = mse(h[:,1:,:], h_hat_supervise[:,:-1,:])

    # Two Moments
    x_m, x_v = tf.nn.moments(x, axes=[0])
    x_hat_m, x_hat_v = tf.nn.moments(x_hat, axes=[0])
    g_loss_v1 = tf.reduce_mean(tf.abs(tf.sqrt(x_hat_v + 1e-6) - tf.sqrt(x_v + 1e-6)))
    g_loss_v2 = tf.reduce_mean(tf.abs(x_hat_m - x_m))
    g_loss_v = g_loss_v1 + g_loss_v2

    g_loss = g_loss_u + gamma * g_loss_u_e + 100 * tf.sqrt(g_loss_s) + 100 * g_loss_v
    return g_loss, g_loss_s

def embedder_loss(x, x_tilde, g_loss_s):
    e_loss_t0 = mse(x, x_tilde)
    e_loss0 = 10 * tf.sqrt(e_loss_t0)
    e_loss = e_loss0 + 0.1 * g_loss_s
    return e_loss, e_loss0

# -----------------------------------------------------------------------------
# Training Function (TimeGAN main)
# -----------------------------------------------------------------------------
def timegan(ori_data, parameters):
    print("✅ TensorFlow 2.x TimeGAN starting...")
    
    # Unpack parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    gamma = 1
    z_dim = ori_data.shape[-1]

    # Normalize
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    ori_time, max_seq_len = extract_time(ori_data)
    no, seq_len, dim = np.asarray(ori_data).shape

    # Instantiate models
    embedder = Embedder(module_name, hidden_dim, num_layers)
    recovery = Recovery(module_name, hidden_dim, dim, num_layers)
    generator = Generator(module_name, hidden_dim, num_layers)
    supervisor = Supervisor(module_name, hidden_dim, num_layers)
    discriminator = Discriminator(module_name, hidden_dim, num_layers)

    # Optimizers
    opt_e = tf.keras.optimizers.Adam()
    opt_g = tf.keras.optimizers.Adam()
    opt_d = tf.keras.optimizers.Adam()

    # -----------------------------------------------------------------------------
    # 1. Embedding Network Training
    print("Start Embedding Network Training...")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        with tf.GradientTape() as tape:
            h = embedder(X_mb)
            x_tilde = recovery(h)
            e_loss, e_loss0 = embedder_loss(X_mb, x_tilde, 0)
        grads = tape.gradient(e_loss0, embedder.trainable_variables + recovery.trainable_variables)
        opt_e.apply_gradients(zip(grads, embedder.trainable_variables + recovery.trainable_variables))

        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(e_loss0.numpy()),4)}")

    # -----------------------------------------------------------------------------
    # 2. Supervised Loss Training
    print("Start Training with Supervised Loss Only...")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        with tf.GradientTape() as tape:
            e_hat = generator(Z_mb)
            h_hat_supervise = supervisor(e_hat)
            g_loss_s = mse(embedder(X_mb)[:,1:,:], h_hat_supervise[:,:-1,:])
        grads = tape.gradient(g_loss_s, generator.trainable_variables + supervisor.trainable_variables)
        opt_g.apply_gradients(zip(grads, generator.trainable_variables + supervisor.trainable_variables))

        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(g_loss_s.numpy()),4)}")

    # -----------------------------------------------------------------------------
    # 3. Joint Training
    print("Start Joint Training...")
    for itt in range(iterations):
        for _ in range(2):  # Generator twice
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            with tf.GradientTape(persistent=True) as tape:
                h = embedder(X_mb)
                x_tilde = recovery(h)
                e_hat = generator(Z_mb)
                h_hat = supervisor(e_hat)
                h_hat_supervise = supervisor(h)
                x_hat = recovery(h_hat)
                y_fake = discriminator(h_hat)
                y_real = discriminator(h)
                y_fake_e = discriminator(e_hat)

                g_loss, g_loss_s = generator_loss(y_fake, y_fake_e, h, h_hat_supervise, X_mb, x_hat, gamma)
                e_loss, e_loss0 = embedder_loss(X_mb, x_tilde, g_loss_s)

            # Generator + Embedder updates
            grads_g = tape.gradient(g_loss, generator.trainable_variables + supervisor.trainable_variables)
            opt_g.apply_gradients(zip(grads_g, generator.trainable_variables + supervisor.trainable_variables))
            grads_e = tape.gradient(e_loss, embedder.trainable_variables + recovery.trainable_variables)
            opt_e.apply_gradients(zip(grads_e, embedder.trainable_variables + recovery.trainable_variables))

        # Discriminator
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        with tf.GradientTape() as tape:
            h = embedder(X_mb)
            e_hat = generator(Z_mb)
            h_hat = supervisor(e_hat)
            y_fake = discriminator(h_hat)
            y_real = discriminator(h)
            y_fake_e = discriminator(e_hat)
            d_loss = discriminator_loss(y_real, y_fake, y_fake_e, gamma)
        if d_loss.numpy() > 0.15:
            grads_d = tape.gradient(d_loss, discriminator.trainable_variables)
            opt_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, d_loss: {np.round(d_loss.numpy(),4)}, g_loss_s: {np.round(np.sqrt(g_loss_s.numpy()),4)}")

    # -----------------------------------------------------------------------------
    # Synthetic Data Generation
    print("Generating synthetic data...")
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    h_hat = supervisor(generator(Z_mb))
    generated_data_curr = recovery(h_hat).numpy()

    generated_data = []
    for i in range(no):
        generated_data.append(generated_data_curr[i, :ori_time[i], :])

    generated_data = np.array(generated_data)
    generated_data = generated_data * max_val + min_val
    return generated_data
