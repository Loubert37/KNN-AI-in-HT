# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.stride_tricks import as_strided

# ==============================================================================
# Convolution Forward (Optimisé via im2col + multiplication matricielle BLAS)
# ==============================================================================

def _im2col(img, m, s_h, s_w):
    batch_size, h_pad, w_pad, c = img.shape
    h_out = (h_pad - m) // s_h + 1
    w_out = (w_pad - m) // s_w + 1
    sB, s0, s1, s2 = img.strides

    fenetre = as_strided(
        img,
        shape=(batch_size, h_out, w_out, m, m, c),
        strides=(sB, s0 * s_h, s1 * s_w, s0, s1, s2)
    )
    cols = np.ascontiguousarray(fenetre).reshape(batch_size * h_out * w_out, m * m * c)
    return cols, h_out, w_out


def convolution_batch(I, K, DEBUG=False, stride=1):
    batch_size, _, _, c = I.shape
    C_out, m, _, _ = K.shape
    p = m // 2
    img = np.pad(I, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)

    cols, h_out, w_out = _im2col(img, m, s_h, s_w)
    K_col = K.reshape(C_out, m * m * c).T

    out = cols @ K_col
    return out.reshape(batch_size, h_out, w_out, C_out)


def conv_forward(I, K, B, DEBUG=False, stride=1):
    return convolution_batch(I, K, DEBUG, stride=stride) + B


# ==============================================================================
# Convolution Backward (Optimisé BLAS)
# ==============================================================================

def conv_backward(delta_apres, I, K, stride=1):
    batch_size, h_in, w_in, c_in = I.shape
    C_out, m, _, _ = K.shape
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)

    p = m // 2
    img = np.pad(I, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
    cols, h_out, w_out = _im2col(img, m, s_h, s_w)

    delta_flat = delta_apres.reshape(-1, C_out)

    dK = (delta_flat.T @ cols).reshape(C_out, m, m, c_in)
    dB = np.sum(delta_flat, axis=0)

    if s_h == 1 and s_w == 1:
        K_rot = np.ascontiguousarray(K[:, ::-1, ::-1, :].transpose(3, 1, 2, 0))
        K_rot_col = K_rot.reshape(c_in, m * m * C_out).T
        delta_pad = np.pad(delta_apres, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        d_cols, _, _ = _im2col(delta_pad, m, 1, 1)
        delta_in = (d_cols @ K_rot_col).reshape(batch_size, h_in, w_in, c_in)
    else:
        delta_pad = np.zeros((batch_size, h_in + 2 * p, w_in + 2 * p, c_in), dtype=delta_apres.dtype)
        K_col = K.reshape(C_out, m * m * c_in)
        grad_windows = (delta_flat @ K_col).reshape(batch_size, h_out, w_out, m, m, c_in)
        for i in range(m):
            for j in range(m):
                delta_pad[:, i:i + h_out * s_h:s_h, j:j + w_out * s_w:s_w, :] += grad_windows[:, :, :, i, j, :]
        delta_in = delta_pad[:, p:-p, p:-p, :] if p > 0 else delta_pad

    return delta_in, dK, dB


# ==============================================================================
# Batch Normalization 2D (Forward & Backward)
# ==============================================================================

def batchnorm2d_forward(X, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    if training:
        mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
        var = np.var(X, axis=(0, 1, 2), keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        X_hat = (X - mean) * std_inv

        running_mean[:] = momentum * running_mean + (1.0 - momentum) * mean.squeeze()
        running_var[:] = momentum * running_var + (1.0 - momentum) * var.squeeze()

        out = gamma * X_hat + beta
        cache = (X_hat, std_inv, gamma)
        return out, cache
    else:
        mean = running_mean.reshape(1, 1, 1, -1)
        var = running_var.reshape(1, 1, 1, -1)
        X_hat = (X - mean) / np.sqrt(var + eps)
        return gamma * X_hat + beta, None


def batchnorm2d_backward(delta_apres, cache):
    X_hat, std_inv, gamma = cache
    N = delta_apres.shape[0] * delta_apres.shape[1] * delta_apres.shape[2]

    dgamma = np.sum(delta_apres * X_hat, axis=(0, 1, 2))
    dbeta = np.sum(delta_apres, axis=(0, 1, 2))

    dX_hat = delta_apres * gamma
    sum_dX_hat = np.sum(dX_hat, axis=(0, 1, 2), keepdims=True)
    sum_dX_hat_X_hat = np.sum(dX_hat * X_hat, axis=(0, 1, 2), keepdims=True)

    delta_in = (std_inv / N) * (N * dX_hat - sum_dX_hat - X_hat * sum_dX_hat_X_hat)
    return delta_in, dgamma, dbeta


# ==============================================================================
# Global Average Pooling (GAP)
# ==============================================================================

def GAP_for(volume):
    return np.mean(volume, axis=(1, 2))


def GAP_back(delta_l, input_shape):
    Batch, H, W, C = input_shape
    delta_reshaped = delta_l[:, None, None, :]
    return np.broadcast_to(delta_reshaped / np.float32(H * W), input_shape)


# ==============================================================================
# Pooling
# ==============================================================================

def pooling_for(I, pool_size):
    batch_size, h, w, c = I.shape
    m, n = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)

    pad_h = (m - h % m) % m
    pad_w = (n - w % n) % n
    if pad_h > 0 or pad_w > 0:
        I_pad = np.pad(I, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:
        I_pad = I

    _, h_pad, w_pad, _ = I_pad.shape
    h_out, w_out = h_pad // m, w_pad // n
    sB, s0, s1, s2 = I_pad.strides

    nouv_shape = (batch_size, h_out, w_out, m, n, c)
    nouv_strides = (sB, s0 * m, s1 * n, s0, s1, s2)

    fenetre = as_strided(I_pad, shape=nouv_shape, strides=nouv_strides)
    max_fenetre = np.max(fenetre, axis=(-3, -2))

    masque_6d = (fenetre == max_fenetre[:, :, :, None, None, :])
    masque_pad = masque_6d.transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, h_pad, w_pad, c)
    cache = masque_pad[:, :h, :w, :]

    return max_fenetre, cache


def pooling_back(delta, masque, shape_origine, pool_size):
    m, n = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
    _, H, W, _ = shape_origine
    h_lim, w_lim = (H // m) * m, (W // n) * n

    delta_repeat = np.repeat(np.repeat(delta, m, axis=1), n, axis=2)
    delta_final = np.zeros(shape_origine, dtype=delta.dtype)
    delta_final[:, :h_lim, :w_lim, :] = delta_repeat * masque

    return delta_final


# ==============================================================================
# Fonctions d'Activation (Forward & Backward)
# ==============================================================================

def sigmoid_for(Z):
    Z_clipped = np.clip(Z, -80.0, 80.0)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-Z_clipped))


def sigmoid_back(delta_apres, A):
    return delta_apres * A * (np.float32(1.0) - A)


def softmax_for(Z):
    e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=-1, keepdims=True)


def softmax_back(delta_apres, S):
    sum_ds = np.sum(delta_apres * S, axis=-1, keepdims=True)
    return S * (delta_apres - sum_ds)


def gelu_for(Z):
    c1 = np.float32(np.sqrt(2.0 / np.pi))
    c2 = np.float32(0.044715)
    return np.float32(0.5) * Z * (np.float32(1.0) + np.tanh(c1 * (Z + c2 * Z**3)))


def gelu_back(delta_apres, Z):
    c1 = np.float32(np.sqrt(2.0 / np.pi))
    c2 = np.float32(0.044715)
    cdf = np.float32(0.5) * (np.float32(1.0) + np.tanh(c1 * (Z + c2 * Z**3)))
    pdf = np.exp(np.float32(-0.5) * Z**2) / np.float32(np.sqrt(2.0 * np.pi))
    dZ = cdf + Z * pdf * (np.float32(1.0) + np.float32(3.0) * c2 * Z**2)
    return delta_apres * dZ


def leaky_relu_for(Z, alpha=0.01):
    alpha_f32 = np.float32(alpha)
    return np.where(Z > 0, Z, Z * alpha_f32)


def leaky_relu_back(delta_apres, Z, alpha=0.01):
    dZ = np.ones_like(Z)
    dZ[Z <= 0] = np.float32(alpha)
    return delta_apres * dZ


# ==============================================================================
# Dense / Fully-Connected Layer Operations
# ==============================================================================

def dense_forward(X, W, B):
    return np.dot(X, W) + B


def dense_backward(delta_apres, X, W):
    dW = np.dot(X.T, delta_apres)
    dB = np.sum(delta_apres, axis=0)
    delta_in = np.dot(delta_apres, W.T)
    return delta_in, dW, dB