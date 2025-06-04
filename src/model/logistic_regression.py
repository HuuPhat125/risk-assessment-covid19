import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=10000, threshold=0.5,
                 penalty=None, C=1.0, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.verbose = verbose

        # Validate regularization parameters
        if penalty not in [None, 'l1', 'l2']:
            raise ValueError("Penalty must be 'l1', 'l2', or None.")
        if penalty is not None and C <= 0:
            raise ValueError("Regularization parameter must be positive.")

        self.penalty = penalty
        self.C = C

        # Model parameters
        self.weights = None
        self.bias = None

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """Compute the binary cross-entropy loss."""
        epsilon = 1e-9
        bce_loss = -np.mean(y_true * np.log(y_pred + epsilon) +
                            (1 - y_true) * np.log(1 - y_pred + epsilon))

        reg_loss = 0
        if self.penalty == 'l1':
            reg_loss = self.C * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            reg_loss = self.C * np.sum(self.weights ** 2)
        elif self.penalty is not None:
            raise ValueError(
                "Invalid penalty type. Use 'l1', 'l2', or None.")

        return bce_loss + reg_loss

    def predict_proba(self, X):
        """Compute the linear combination of inputs and weights."""

        if self.weights is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")

        z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(z)
        return A

    def predict(self, X):
        """Predict binary labels for the input data."""
        y_pred_proba = self.predict_proba(X)
        y_class = (y_pred_proba >= self.threshold).astype(int)
        return y_class

    def compute_accuracy(self, y_true, y_pred_proba):
        """Compute the accuracy of the model predictions."""
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

    def compute_gradients(self, X, y, y_pred):
        """Compute the gradients for weights and bias."""
        n_samples = X.shape[0]

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Apply regularization
        if self.penalty == 'l1':
            dw += self.C * np.sign(self.weights)
        elif self.penalty == 'l2':
            dw += 2 * self.C * self.weights

        return dw, db

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping=False, patience=10):
        """Fit the model to the training data."""

        # Training input validation
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."
        assert np.all((y == 0) | (y == 1)), "y must be binary (0 or 1)."

        # Validate validation data if provided
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            assert X_val.shape[0] == y_val.shape[0], "X_val and y_val must have the same number of samples."
            assert np.all((y_val == 0) | (y_val == 1)
                          ), "y_val must be binary (0 or 1)."

        n_samples, n_features = X.shape

        # Initialize weights and bias
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, n_features)
        self.bias = 0

        # Initialize best weights and bias for early stopping
        self.best_weights = None
        self.best_bias = None

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for iter in range(self.max_iter):
            # Forward pass
            y_pred_proba = self.predict_proba(X)

            # Compute train loss and accuracy
            train_loss = self.compute_loss(y, y_pred_proba)
            train_accuracy = self.compute_accuracy(y, y_pred_proba)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred_proba)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Validation step
            if has_val:
                y_val_pred_proba = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred_proba)
                val_accuracy = self.compute_accuracy(y_val, y_val_pred_proba)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                # Early stopping check
                if early_stopping:
                    if val_loss <= best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.best_weights = self.weights.copy()
                        self.best_bias = self.bias
                    else:
                        patience_counter += 1
                        if self.verbose:
                            print(
                                f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"Early stopping at iteration {iter + 1}.")
                        break

            # Print loss every 200 iterations
            if self.verbose and (iter % 200 == 0 or iter == self.max_iter - 1):
                if has_val:
                    print(
                        f"Iteration {iter}/{self.max_iter}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
                        f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
                else:
                    print(
                        f"Iteration {iter}/{self.max_iter}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")

        self.weights = self.best_weights if self.best_weights is not None else self.weights
        self.bias = self.best_bias if self.best_bias is not None else self.bias

        return self

    def get_training_history(self):
        """Return the history of losses and accuracies during training."""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
