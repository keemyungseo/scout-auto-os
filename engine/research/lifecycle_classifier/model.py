"""Multinomial logistic regression (numpy) for lifecycle label probabilities."""

from __future__ import annotations

import math

import numpy as np


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


class MultinomialLifecycleClassifier:
    """Entry-time multiclass classifier — outputs lifecycle probabilities only."""

    def __init__(
        self,
        class_names: list[str],
        max_iter: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.class_names = list(class_names)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = len(self.class_names)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Xs = (X - self.mean_) / self.std_

        counts = np.bincount(y, minlength=k).astype(float)
        counts[counts == 0] = 1.0
        class_weight = n / (k * counts)

        W = rng.normal(0, 0.01, size=(d, k))
        b = np.zeros(k, dtype=float)
        Y = _one_hot(y, k)

        lr = self.learning_rate
        loss_history: list[float] = []

        for _ in range(self.max_iter):
            logits = Xs @ W + b
            probs = _softmax_rows(logits)
            sample_w = class_weight[y][:, None]
            grad_logits = (probs - Y) * sample_w
            grad_W = (Xs.T @ grad_logits) / n + self.l2 * W
            grad_b = grad_logits.mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b

            ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12) * class_weight[y])
            loss_history.append(float(ce + 0.5 * self.l2 * np.sum(W * W)))

        self.weights_ = W
        self.bias_ = b
        train_acc = float((self.predict(X) == y).mean())
        return {
            "train_accuracy": round(train_acc, 4),
            "final_loss": round(loss_history[-1], 6) if loss_history else None,
            "class_counts": {self.class_names[i]: int(c) for i, c in enumerate(counts)},
        }

    def _transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None and self.bias_ is not None
        logits = self._transform(X) @ self.weights_ + self.bias_
        return _softmax_rows(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def proba_dicts(self, X: np.ndarray) -> list[dict[str, float]]:
        probs = self.predict_proba(X)
        out: list[dict[str, float]] = []
        for row in probs:
            out.append(
                {self.class_names[i]: round(float(row[i]) * 100.0, 2) for i in range(len(self.class_names))},
            )
        return out
