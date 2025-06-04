from collections import Counter, defaultdict
import numpy as np
from math import log, exp, pi, sqrt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)


class ContinuousFeatureVectors:
    def __init__(self):
        self.stats = {}
        self.mean_var = {}

    def add(self, label, index, value):
        if label not in self.stats:
            self.stats[label] = {}
        if index not in self.stats[label]:
            self.stats[label][index] = []
        self.stats[label][index].append(value)

    def set_mean_variance(self, label):
        if label not in self.mean_var:
            self.mean_var[label] = {}

        if label in self.stats:
            for index, values in self.stats[label].items():
                n = len(values)
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / \
                    (n - 1) if n > 1 else 1e-6
                self.mean_var[label][index] = (mean, variance)

    def probability(self, label, index):
        if label not in self.mean_var or index not in self.mean_var[label]:
            mean, var = 0.0, 1.0
        else:
            mean, var = self.mean_var[label][index]

        if var == 0:
            var = 1e-6

        def prob(x):
            return (1.0 / sqrt(2 * pi * var)) * exp(-(x - mean) ** 2 / (2 * var))
        return prob


class DiscreteFeatureVectors:
    def __init__(self, use_smoothing=True):
        self.use_smoothing = use_smoothing
        self.counts = {}
        self.feature_values = {}

    def add(self, label, index, value):
        if label not in self.counts:
            self.counts[label] = {}
        if index not in self.counts[label]:
            self.counts[label][index] = {}
        if value not in self.counts[label][index]:
            self.counts[label][index][value] = 0

        self.counts[label][index][value] += 1

        if index not in self.feature_values:
            self.feature_values[index] = set()
        self.feature_values[index].add(value)

    def probability(self, label, index, value, total):
        count = 0
        if (label in self.counts and
            index in self.counts[label] and
                value in self.counts[label][index]):
            count = self.counts[label][index][value]

        if self.use_smoothing:
            k = len(self.feature_values.get(index, set()))
            return (count + 1) / (total + k)
        else:
            return count / total if total > 0 else 1e-6


class NaiveBayes:
    def __init__(self, continuous_features: set = None, use_smoothing=True):
        if isinstance(continuous_features, (list, set, tuple)):
            self.continuous_features = set(continuous_features)
        else:
            self.continuous_features = set()

        self.use_smoothing = use_smoothing

        self.label_counts = Counter()
        self.priors = {}
        self.discrete_vector = DiscreteFeatureVectors(use_smoothing)
        self.continuous_vector = ContinuousFeatureVectors()

        self._is_fitted = False

    def fit(self, X, y):
        for xi, label in zip(X, y):
            self.label_counts[label] += 1
            for idx, value in enumerate(xi):
                if int(idx) in self.continuous_features:
                    self.continuous_vector.add(label, idx, value)
                else:
                    self.discrete_vector.add(label, idx, value)

        total = len(y)
        self.priors = {label: count / total for label,
                       count in self.label_counts.items()}

        for label in self.label_counts:
            self.continuous_vector.set_mean_variance(label)

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("NaiveBayes model is not fitted.")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return [self._predict_single(xi) for xi in X]

    def _predict_single(self, x):
        log_likelihood = {label: log(prior)
                          for label, prior in self.priors.items()}

        for label in self.label_counts:
            for idx, value in enumerate(x):
                prob = self._get_feature_probability(idx, value, label)
                log_likelihood[label] += log(prob if prob > 0 else 1e-6)

        return max(log_likelihood, key=log_likelihood.get)

    def _get_feature_probability(self, idx, value, label):
        if int(idx) in self.continuous_features:
            if not self._is_fitted:
                raise RuntimeError("NaiveBayes model is not fitted.")
            return self.continuous_vector.probability(label, idx)(value)
        return self.discrete_vector.probability(label, idx, value, self.label_counts[label])

    def compute_metrics(self, y_true, y_pred):
        if not self._is_fitted:
            raise RuntimeError("NaiveBayes model is not fitted.")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report
        }
