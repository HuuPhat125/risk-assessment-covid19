from model.decision_tree import DecisionTree
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: int | None = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        max_features: int | float | None = 'sqrt',
        random_state: int = 0,
        max_leaf_nodes: int | None = None,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        init_log: bool = True
    ):
        if criterion in ['gini', 'entropy', 'log_loss']:
            self.criterion = criterion
        else:
            raise ValueError("Invalid criterion")

        if not (isinstance(min_samples_split, int) and min_samples_split >= 2):
            raise ValueError("min_samples_split must be int >= 2")

        if not ((isinstance(min_samples_leaf, int) and min_samples_leaf >= 1) or
                (isinstance(min_samples_leaf, float) and 0.0 < min_samples_leaf <= 0.5)):
            raise ValueError("Invalid min_samples_leaf")
        
                # Kiểm tra và xác thực max_features
        valid_max_features = [None, 'sqrt', 'log2']
        if not (max_features is None or isinstance(max_features, (int, float)) or max_features in valid_max_features):
            raise ValueError("max_features must be int, float, 'sqrt', 'log2' or None")
        
        if isinstance(max_features, int) and max_features <= 0:
            raise ValueError("max_features int must be positive")
        
        if isinstance(max_features, float) and not (0.0 < max_features <= 1.0):
            raise ValueError("max_features float must be in (0.0, 1.0]")
    
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        self.trees = []
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        if init_log: 
            print("RandomForestScratch initialized with parameters:")
            print(f"  n_estimators     = {self.n_estimators}")
            print(f"  criterion        = {self.criterion}")
            print(f"  max_depth        = {self.max_depth}")
            print(f"  min_samples_split = {self.min_samples_split}")
            print(f"  min_samples_leaf  = {self.min_samples_leaf}")
            print(f"  max_features      = {self.max_features}")
            print(f"  bootstrap         = {self.bootstrap}")
            print(f"  max_samples       = {self.max_samples}")
            print(f"  random_state      = {self.random_state}")
            print(f"  max_leaf_nodes    = {self.max_leaf_nodes}")

    def _determine_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'")
        else:
            raise ValueError("Invalid max_features parameter")

    def _determine_max_samples(self, n_total):
        if self.max_samples is None:
            return n_total
        elif isinstance(self.max_samples, int):
            return min(n_total, self.max_samples)
        elif isinstance(self.max_samples, float):
            return max(1, int(self.max_samples * n_total))
        else:
            raise ValueError("max_samples must be int, float, or None")

    def _sample_data(self, X, y):
        n_samples = X.shape[0]
        n_draws = self._determine_max_samples(n_samples)

        if self.bootstrap:
            indices = np.random.choice(n_samples, n_draws, replace=True)
        else:
            indices = np.random.choice(n_samples, n_draws, replace=False)

        return X[indices], y[indices]

    def fit(self, X, y):
        # Convert data to numpy
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()

        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        max_feats = self._determine_max_features(self.n_features)

        self.trees = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self._sample_data(X, y)

            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feats,
                random_state=self.random_state + i, 
                init_log=False
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        predictions = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_estimators, n_samples)

        y_pred = []
        for sample_preds in predictions.T:
            counts = Counter(sample_preds)
            y_pred.append(counts.most_common(1)[0][0])
        return np.array(y_pred)

    
    def score(self, X, y, average: str = 'weighted', report_dict: bool = True) -> dict:
        """
        Returns a dictionary of classification metrics.
        For multiclass, set average to 'macro', 'micro', or 'weighted'.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average=average, zero_division=0),
            'recall': recall_score(y, y_pred, average=average, zero_division=0),
            'f1': f1_score(y, y_pred, average=average, zero_division=0),
            'classification_report': classification_report(
                y, y_pred, output_dict=report_dict, zero_division=0
            )
        }
        return metrics       