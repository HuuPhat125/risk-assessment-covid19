import os
import numpy as np
import pandas as pd
from collections import Counter
import random
import math
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

class TreeNode:
    def __init__(self):
        self.split_feature = None
        self.feature_value = None
        self.proba = None
        self.left_child: TreeNode = None
        self.right_child: TreeNode = None

    def is_leaf(self):
        return self.proba is not None

class DecisionTree:
    def __init__(
        self,
        criterion: str = 'gini',
        splitter: str = 'best',
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        init_log: bool = True,
    ):
        if criterion in ['gini', 'entropy', 'log_loss']:
            self.criterion = criterion
        else:
            raise ValueError("Invalid criterion")

        if splitter in ['best', 'random']:
            self.splitter = splitter
        else:
            raise ValueError("Invalid splitter")

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

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.leaf_count = 0

        self.root = None

        # In ra các tham số của mô hình sau khi khởi tạo
        if init_log: 
            print("DecisionTreeScratch initialized with parameters:")
            print(f"  criterion          = {self.criterion}")
            print(f"  splitter           = {self.splitter}")
            print(f"  max_depth          = {self.max_depth}")
            print(f"  min_samples_split  = {self.min_samples_split}")
            print(f"  min_samples_leaf   = {self.min_samples_leaf}")
            print(f"  max_features       = {self.max_features}")
            print(f"  random_state       = {self.random_state}")
            print(f"  max_leaf_nodes     = {self.max_leaf_nodes}")

    def _cal_proba(self, y, num_classes=None):
        y = y.astype(int)
        if num_classes is None:
            num_classes = len(np.unique(y))
        counter = Counter(y)
        proba = np.zeros(num_classes, dtype=float)
        total = len(y)
        for label, count in counter.items():
            proba[label] = count / total
        return proba

    def _gini(self, y, num_classes=None):
        proba = self._cal_proba(y, num_classes)
        return 1.0 - np.sum(proba ** 2)

    def _entropy(self, y, num_classes=None):
        proba = self._cal_proba(y, num_classes)
        return -np.sum(proba[proba > 0] * np.log2(proba[proba > 0]))

    def _log_loss(self, y, num_classes=None):
        proba = self._cal_proba(y, num_classes)
        return -np.sum(proba[proba > 0] * np.log(proba[proba > 0]))

    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y, self.n_classes)
        elif self.criterion == 'entropy':
            return self._entropy(y, self.n_classes)
        elif self.criterion == 'log_loss':
            return self._log_loss(y, self.n_classes)


    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_value = None
        current_impurity = self._impurity(y)
        n_samples, n_features = X.shape

        # Xác định số lượng feature được chọn tại node này
        if self.max_features is None:
            features = list(range(n_features))
        else:
            if isinstance(self.max_features, float):
                # tỷ lệ phần trăm features
                k = max(1, int(self.max_features * n_features))
            elif isinstance(self.max_features, int):
                k = min(self.max_features, n_features)
            elif self.max_features == 'sqrt':
                k = max(1, int(math.sqrt(n_features)))
            elif self.max_features == 'log2':
                k = max(1, int(math.log2(n_features)))
            else:
                raise ValueError("Invalid max_features value")

            rng = np.random.default_rng(self.random_state)
            features = rng.choice(n_features, k, replace=False)

        # Nếu splitter là random, xáo trộn thứ tự feature
        if self.splitter == 'random':
            random.seed(self.random_state)
            features = list(features)
            random.shuffle(features)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                p = float(len(y[left_idx])) / len(y)
                gain = current_impurity - p * self._impurity(y[left_idx]) - (1 - p) * self._impurity(y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold

        return best_feature, best_value

    def _build_tree(self, X, y, depth):
        node = TreeNode()

        if (
            self.max_depth is not None and depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split or
            (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes)
        ):
            node.proba = self._cal_proba(y, self.n_classes)
            self.leaf_count += 1
            return node

        feature, value = self._best_split(X, y)
        if feature is None:
            node.proba = self._cal_proba(y, self.n_classes)
            return node

        node.split_feature = feature
        node.feature_value = value

        left_idx = X[:, feature] <= value
        right_idx = ~left_idx

        node.left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    def fit(self, X, y):
        X, y = self.ensure_numpy(X,y)
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.split_feature] <= node.feature_value:
                node = node.left_child
            else:
                node = node.right_child
        return np.argmax(node.proba)

    def predict(self, X):
        X, y = self.ensure_numpy(X, [])
        return np.array([self.predict_one(x) for x in X])   
    
    def ensure_numpy(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        return X, y

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

