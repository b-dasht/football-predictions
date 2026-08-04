"""A from-scratch PyTorch Neural Network - an exposure exercise (see
docs/PROJECT_OVERVIEW.md), not part of the primary model-comparison
methodology in models.py, and deliberately not using a wrapper library
like skorch: the network definition and the training loop (forward pass,
loss, backward pass, optimizer step) are written by hand, since that's
the part of PyTorch that's genuinely different from scikit-learn's
"call .fit() and it's done."

Each class still exposes the plain scikit-learn interface (.fit/.predict/
.predict_proba, plus get_params/set_params via BaseEstimator) purely so
it can plug into the exact same train_and_save_classifier/regressor
helpers, results log, and comparison plots as every other model - the
PyTorch code itself doesn't know or care that it's wrapped this way.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """One hidden layer + ReLU, matching the scikit-learn MLP's architecture
    (see build_neural_network_classifier in models.py) - same recipe,
    different library, for a clean side-by-side comparison.

    Per scikit-learn's estimator convention, __init__ only stores its
    arguments (nothing else) - the actual network only gets built inside
    fit(), once the number of input features and classes is known.
    """

    def __init__(self, hidden_size: int = 64, epochs: int = 300, lr: float = 0.001, random_state: int = 42):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        # classes_ handles arbitrary label values (e.g. the 2-class {0, 2}
        # encoding) the same way scikit-learn's own classifiers do -
        # np.unique sorts ascending, so column order in predict_proba
        # always matches labels passed to classification_metrics.
        self.classes_ = np.unique(y)
        label_to_index = {label: i for i, label in enumerate(self.classes_)}
        y_indices = torch.tensor([label_to_index[v] for v in np.asarray(y)], dtype=torch.long)

        n_features = X_tensor.shape[1]
        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, len(self.classes_)),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(X_tensor)
            loss = loss_fn(logits, y_indices)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        indices = self.predict_proba(X).argmax(axis=1)
        return self.classes_[indices]


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Same architecture/training-loop approach as TorchMLPClassifier, with
    a single output neuron and no activation on it (goal difference is an
    unbounded continuous value) and MSE loss instead of cross-entropy.
    """

    def __init__(self, hidden_size: int = 64, epochs: int = 300, lr: float = 0.001, random_state: int = 42):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

        n_features = X_tensor.shape[1]
        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            predictions = self.model_(X_tensor)
            loss = loss_fn(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X) -> np.ndarray:
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor).numpy().flatten()
        return predictions
