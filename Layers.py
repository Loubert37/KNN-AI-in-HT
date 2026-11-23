# -*- coding: utf-8 -*-
import numpy as np

from Core_ops import (
    conv_forward, conv_backward,
    batchnorm2d_forward, batchnorm2d_backward,
    pooling_for, pooling_back,
    GAP_for, GAP_back,
    dense_forward, dense_backward,
    leaky_relu_for, leaky_relu_back,
    gelu_for, gelu_back,
    sigmoid_for, sigmoid_back,
    softmax_for, softmax_back
)


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.training = True

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, delta_apres):
        raise NotImplementedError


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.m = kernel_size
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
        self.K = (np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * scale).astype(np.float32)
        self.B = np.zeros(out_channels, dtype=np.float32)
        self.dK = None
        self.dB = None
        self.I_cache = None

    def forward(self, I):
        self.I_cache = I
        return conv_forward(I, self.K, self.B, stride=self.stride)

    def backward(self, delta_apres):
        delta_in, self.dK, self.dB = conv_backward(delta_apres, self.I_cache, self.K, stride=self.stride)
        return delta_in


class BatchNorm2D(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.dgamma = None
        self.dbeta = None
        self.cache = None

    def forward(self, X):
        out, self.cache = batchnorm2d_forward(
            X, self.gamma, self.beta,
            self.running_mean, self.running_var,
            training=self.training, momentum=self.momentum, eps=self.eps
        )
        return out

    def backward(self, delta_apres):
        delta_in, self.dgamma, self.dbeta = batchnorm2d_backward(delta_apres, self.cache)
        return delta_in


class MaxPool2D(Layer):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.mask_cache = None
        self.shape_in_cache = None

    def forward(self, I):
        self.shape_in_cache = I.shape
        out, self.mask_cache = pooling_for(I, self.pool_size)
        return out

    def backward(self, delta_apres):
        return pooling_back(delta_apres, self.mask_cache, self.shape_in_cache, self.pool_size)


class GlobalAvgPool(Layer):
    def __init__(self):
        super().__init__()
        self.shape_in_cache = None

    def forward(self, I):
        self.shape_in_cache = I.shape
        return GAP_for(I)

    def backward(self, delta_apres):
        return GAP_back(delta_apres, self.shape_in_cache)


class Dense(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features) * scale).astype(np.float32)
        self.B = np.zeros(out_features, dtype=np.float32)
        self.dW = None
        self.dB = None
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return dense_forward(X, self.W, self.B)

    def backward(self, delta_apres):
        delta_in, self.dW, self.dB = dense_backward(delta_apres, self.X_cache, self.W)
        return delta_in


class Dropout(Layer):
    def __init__(self, rate=0.3):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, X):
        if not self.training or self.rate <= 0.0:
            return X
        keep_prob = np.float32(1.0 - self.rate)
        self.mask = (np.random.rand(*X.shape) < keep_prob).astype(np.float32) / keep_prob
        return X * self.mask

    def backward(self, delta_apres):
        if not self.training or self.rate <= 0.0:
            return delta_apres
        return delta_apres * self.mask


class Activation(Layer):
    def __init__(self, mode='leaky_relu', alpha=0.01):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.Z_cache = None
        self.A_cache = None

    def forward(self, Z):
        self.Z_cache = Z
        if self.mode == 'leaky_relu':
            return leaky_relu_for(Z, self.alpha)
        elif self.mode == 'gelu':
            return gelu_for(Z)
        elif self.mode == 'sigmoid':
            self.A_cache = sigmoid_for(Z)
            return self.A_cache
        elif self.mode == 'softmax':
            self.A_cache = softmax_for(Z)
            return self.A_cache

    def backward(self, delta_apres):
        if self.mode == 'leaky_relu':
            return leaky_relu_back(delta_apres, self.Z_cache, self.alpha)
        elif self.mode == 'gelu':
            return gelu_back(delta_apres, self.Z_cache)
        elif self.mode == 'sigmoid':
            return sigmoid_back(delta_apres, self.A_cache)
        elif self.mode == 'softmax':
            return softmax_back(delta_apres, self.A_cache)