# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Layers import Conv2D, BatchNorm2D, MaxPool2D, GlobalAvgPool, Dense, Dropout, Activation


class CNN:
    def __init__(self, architecture):
        self.layers = []
        for layer_cfg in architecture:
            layer_type = layer_cfg[0].lower()
            if layer_type == "conv":
                _, in_c, out_c, k_size, stride, use_maxpool = layer_cfg
                self.layers.append(Conv2D(in_c, out_c, kernel_size=k_size, stride=stride))
                if use_maxpool:
                    self.layers.append(MaxPool2D(pool_size=2))
            elif layer_type == "bn":
                _, num_features = layer_cfg
                self.layers.append(BatchNorm2D(num_features))
            elif layer_type == "dense":
                _, in_f, out_f = layer_cfg
                self.layers.append(Dense(in_f, out_f))
            elif layer_type == "gap":
                self.layers.append(GlobalAvgPool())
            elif layer_type == "dropout":
                _, rate = layer_cfg
                self.layers.append(Dropout(rate=rate))
            elif layer_type == "activation":
                _, mode = layer_cfg
                self.layers.append(Activation(mode=mode))

    def set_training(self, mode=True):
        for layer in self.layers:
            layer.training = mode

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, delta_loss):
        delta = delta_loss
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta

    def load_weights(self, filename="best_model_weights.npz"):
        data = np.load(filename)
        for idx, layer in enumerate(self.layers):
            layer_name = f"layer_{idx}_{layer.__class__.__name__}"
            for attr in ['K', 'W', 'B', 'gamma', 'beta', 'running_mean', 'running_var']:
                key = f"{layer_name}_{attr}"
                if hasattr(layer, attr) and key in data:
                    setattr(layer, attr, data[key])
        print(f"\n[+] Poids et statistiques BatchNorm chargés depuis '{filename}'.")


class SGDOptimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                layer.K -= self.lr * layer.dK
                layer.B -= self.lr * layer.dB
            elif isinstance(layer, Dense):
                layer.W -= self.lr * layer.dW
                layer.B -= self.lr * layer.dB
            elif isinstance(layer, BatchNorm2D):
                layer.gamma -= self.lr * layer.dgamma
                layer.beta -= self.lr * layer.dbeta


class AdamOptimizer:
    """AdamW Optimizer avec Weight Decay et support Conv2D, Dense et BatchNorm2D."""
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.model = model
        self.lr = np.float32(lr)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.eps = np.float32(eps)
        self.weight_decay = np.float32(weight_decay)
        self.t = 0
        self.m = {}
        self.v = {}

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2D):
                self.m[idx] = {'K': np.zeros_like(layer.K), 'B': np.zeros_like(layer.B)}
                self.v[idx] = {'K': np.zeros_like(layer.K), 'B': np.zeros_like(layer.B)}
            elif isinstance(layer, Dense):
                self.m[idx] = {'W': np.zeros_like(layer.W), 'B': np.zeros_like(layer.B)}
                self.v[idx] = {'W': np.zeros_like(layer.W), 'B': np.zeros_like(layer.B)}
            elif isinstance(layer, BatchNorm2D):
                self.m[idx] = {'gamma': np.zeros_like(layer.gamma), 'beta': np.zeros_like(layer.beta)}
                self.v[idx] = {'gamma': np.zeros_like(layer.gamma), 'beta': np.zeros_like(layer.beta)}

    def step(self):
        self.t += 1
        bc1 = np.float32(1.0 - self.beta1 ** self.t)
        bc2 = np.float32(1.0 - self.beta2 ** self.t)

        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2D):
                params = [('K', layer.dK, True), ('B', layer.dB, False)]
            elif isinstance(layer, Dense):
                params = [('W', layer.dW, True), ('B', layer.dB, False)]
            elif isinstance(layer, BatchNorm2D):
                params = [('gamma', layer.dgamma, False), ('beta', layer.dbeta, False)]
            else:
                continue

            for param_name, grad, apply_wd in params:
                param = getattr(layer, param_name)
                if apply_wd and self.weight_decay > 0:
                    param -= self.lr * self.weight_decay * param

                self.m[idx][param_name] = self.beta1 * self.m[idx][param_name] + (np.float32(1.0) - self.beta1) * grad
                self.v[idx][param_name] = self.beta2 * self.v[idx][param_name] + (np.float32(1.0) - self.beta2) * (grad ** 2)
                m_hat = self.m[idx][param_name] / bc1
                v_hat = self.v[idx][param_name] / bc2
                setattr(layer, param_name, param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))


class CategoricalCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.probs = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        self.y_true = y_true
        return -np.sum(self.y_true * np.log(self.probs)) / y_pred.shape[0]

    def backward(self):
        N = self.probs.shape[0]
        return -(self.y_true / self.probs) / np.float32(N)


class BinaryCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.probs = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        self.y_true = y_true
        return -np.mean(
            self.y_true * np.log(self.probs) + (1.0 - self.y_true) * np.log(1.0 - self.probs)
        )

    def backward(self):
        N = self.probs.shape[0]
        return (-(self.y_true / self.probs) + (1.0 - self.y_true) / (1.0 - self.probs)) / np.float32(N)


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def backward(self):
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / np.float32(N)


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [],  'val_acc': []
        }

    def save_weights(self, filename="best_model_weights.npz", verbose=True):
        weights_dict = {}
        for idx, layer in enumerate(self.model.layers):
            layer_name = f"layer_{idx}_{layer.__class__.__name__}"
            for attr in ['K', 'W', 'B', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(layer, attr):
                    weights_dict[f"{layer_name}_{attr}"] = getattr(layer, attr)
        np.savez(filename, **weights_dict)
        if verbose:
            print(f"\n[+] Poids sauvegardés dans '{filename}'.")

    def fit(self, X_train, y_train, epochs=10, batch_size=32, X_val=None, y_val=None, augment=True, lr_decay_step=12):
        num_samples = X_train.shape[0]
        total_batches = (num_samples + batch_size - 1) // batch_size
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.set_training(True)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            running_loss = 0.0
            correct_predictions = 0

            for step, start_idx in enumerate(range(0, num_samples, batch_size), 1):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx].copy()
                y_batch = y_shuffled[start_idx:end_idx]

                if augment:
                    if np.random.rand() > 0.5:
                        X_batch = X_batch[:, :, ::-1, :]
                    if np.random.rand() > 0.5:
                        X_batch = X_batch[:, ::-1, :, :]
                    k_rot = np.random.randint(0, 4)
                    if k_rot > 0:
                        X_batch = np.ascontiguousarray(np.rot90(X_batch, k=k_rot, axes=(1, 2)))

                y_pred = self.model.forward(X_batch)
                loss = self.criterion.forward(y_pred, y_batch)
                running_loss += loss * (end_idx - start_idx)

                if y_batch.shape[1] == 4:
                    correct_predictions += np.sum(np.sqrt(np.sum((y_pred - y_batch)**2, axis=1)))
                else:
                    preds = np.argmax(y_pred, axis=1)
                    targets = np.argmax(y_batch, axis=1)
                    correct_predictions += np.sum(preds == targets)

                delta_loss = self.criterion.backward()
                self.model.backward(delta_loss)
                self.optimizer.step()

                print(f"\rEpoch [{epoch}/{epochs}] - Batch {step}/{total_batches}...", end="", flush=True)

            epoch_loss = running_loss / num_samples
            epoch_metric = correct_predictions / num_samples

            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_metric)

            val_info = ""
            saved_tag = ""
            if X_val is not None and y_val is not None:
                val_loss, val_metric = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_metric)

                metric_label = "coord_err" if y_val.shape[1] == 4 else "acc"
                val_info = f" - val_loss: {val_loss:.4f} - val_{metric_label}: {val_metric:.4f}"

                if val_metric > best_val_acc:
                    best_val_acc = val_metric
                    self.save_weights("best_model_weights.npz", verbose=False)
                    saved_tag = "  <-- [BEST SAVED]"

            metric_label = "coord_err" if y_train.shape[1] == 4 else "acc"
            print(f"\rEpoch [{epoch}/{epochs}] - loss: {epoch_loss:.4f} - {metric_label}: {epoch_metric:.4f}{val_info}{saved_tag}      ")

            if lr_decay_step and epoch % lr_decay_step == 0:
                self.optimizer.lr *= np.float32(0.5)

        print("\n==================================================")
        print(f"  TRAINING COMPLETED | Meilleure Val Accuracy: {best_val_acc * 100:.2f}%")
        print("==================================================")

    def evaluate(self, X, y, batch_size=64):
        self.model.set_training(False)
        num_samples = X.shape[0]
        total_loss = 0.0
        correct = 0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_b = X[start_idx:end_idx]
            y_b = y[start_idx:end_idx]

            y_pred = self.model.forward(X_b)
            loss = self.criterion.forward(y_pred, y_b)
            total_loss += loss * (end_idx - start_idx)

            if y_b.shape[1] == 4:
                correct += np.sum(np.sqrt(np.sum((y_pred - y_b)**2, axis=1)))
            else:
                correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_b, axis=1))

        self.model.set_training(True)
        return total_loss / num_samples, correct / num_samples

    def plot_metrics(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss')
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-o', label='Test/Val Loss')
        plt.title('Loss Curve (Overfitting Evaluation)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-o', label='Train Accuracy')
        if self.history['val_acc']:
            plt.plot(epochs, self.history['val_acc'], 'r-o', label='Test/Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()