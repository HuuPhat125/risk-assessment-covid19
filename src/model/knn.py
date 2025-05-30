import numpy as np 
import pandas as pd 
from collections import Counter
from typing import Callable
from tqdm import tqdm
from scipy.spatial import KDTree

class KNeighbors: 
    def __init__(
        self, 
        n_neighbors: int = 5, 
        weights: str | Callable | None = 'uniform', 
        algorithm: str = 'auto',  
        p: float = 2, 
        metric: str | Callable = 'minkowski', 
        init_log: bool = True, 
        batch_size: int = 100,
    ):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        
        if not (weights in ['uniform', 'distance', None] or callable(weights)):
            raise ValueError("weights must be 'uniform', 'distance', a callable, or None.")

        if algorithm not in ['auto', 'kd_tree', 'brute']:
            raise ValueError("algorithm must be one of: 'auto', 'kd_tree', 'brute'.")

        if p <= 0:
            raise ValueError("p must be a positive number.") 

        if not (isinstance(metric, str) or callable(metric)):
            raise ValueError("metric must be a string or a callable.")
        
        self.n_neighbors = n_neighbors 
        self.weights = weights
        self.algorithm = algorithm 
        self.p = p 
        self.metric = metric 
        self.batch_size = batch_size

        # Dùng KDTree nếu phù hợp
        self.use_kdtree = (
            algorithm == 'kd_tree' or
            (
                algorithm == 'auto' and 
                isinstance(metric, str) and 
                metric == 'minkowski' and 
                (p == 1 or p == 2)
            )
        )

        if init_log:
            print("KNeighbors initialized with:")
            print(f"  n_neighbors = {self.n_neighbors}")
            print(f"  weights     = {self.weights}")
            print(f"  algorithm   = {self.algorithm}")
            print(f"  p           = {self.p}")
            print(f"  metric      = {self.metric}") 
            print(f"  batch_size  = {self.batch_size}") 
            print(f"  use_kdtree  = {self.use_kdtree}")

    def _compute_batch_distances(self, X_batch):
        if self.metric == 'minkowski':
            return np.sum(np.abs(X_batch[:, None, :] - self.X_train[None, :, :]) ** self.p, axis=2) ** (1 / self.p)
        elif callable(self.metric):
            return np.array([[self.metric(x, x_train) for x_train in self.X_train] for x in X_batch])
        else:
            raise ValueError("Unsupported metric")
        
    def _vote(self, neighbor_indices, neighbor_distances):
        labels = self.y_train[neighbor_indices]
        if self.weights == 'uniform' or self.weights is None:
            return Counter(labels).most_common(1)[0][0]
        elif self.weights == 'distance':
            label_weights = {}
            for label, dist in zip(labels, neighbor_distances):
                weight = 1 / (dist + 1e-5)  # avoid divide by zero
                label_weights[label] = label_weights.get(label, 0) + weight
            return max(label_weights.items(), key=lambda item: item[1])[0]
        elif callable(self.weights):
            return self.weights(neighbor_distances)
        else:
            raise ValueError("Invalid weights setting")

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

        if self.use_kdtree:
            self.kdtree = KDTree(self.X_train)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        predictions = []

        for i in tqdm(range(0, n_samples, self.batch_size), desc="Predicting (batched)"):
            X_batch = X[i:i + self.batch_size]

            if self.use_kdtree:
                dists, indices = self.kdtree.query(X_batch, k=self.n_neighbors, p=self.p)
                # Nếu chỉ có 1 neighbor thì output là 1D
                if self.n_neighbors == 1:
                    dists = dists[:, None]
                    indices = indices[:, None]
            else:
                distances = self._compute_batch_distances(X_batch)
                indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
                dists = np.take_along_axis(distances, indices, axis=1)

            for j in range(X_batch.shape[0]):
                predictions.append(self._vote(indices[j], dists[j]))

        return np.array(predictions)
