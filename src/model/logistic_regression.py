import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier.
    Parameters:
        learning_rate (float): The learning rate for gradient descent.
        epochs (int): The number of iterations for training.
        weights (float): The weights of the model, initialized to None.
        bias (float): The bias term, initialized to None.
        losses (list): List to store the loss values during training.
        accuracies (list): List to store the accuracy values during training.
    """

    def __init__(self, learning_rate=0.001, epochs=1000, threshold=0.5,
                 regularization=None, reg_param=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

        # Validate regularization parameters
        if regularization not in [None, 'l1', 'l2']:
            raise ValueError("Regularization must be 'l1', 'l2', or None.")
        if regularization is not None and reg_param <= 0:
            raise ValueError("Regularization parameter must be positive.")

        self.regularization = regularization
        self.reg_param = reg_param

        # Model parameters
        self.weights = None
        self.bias = None
        self.best_weights = None
        self.best_bias = None

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
        if self.regularization == 'l1':
            reg_loss = self.reg_param * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_loss = self.reg_param * np.sum(self.weights ** 2)
        elif self.regularization is not None:
            raise ValueError(
                "Invalid regularization type. Use 'l1', 'l2', or None.")

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
        if self.regularization == 'l1':
            dw += self.reg_param * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.reg_param * self.weights

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

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
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
                        print(
                            f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}.")
                        break

            # Print loss every 100 iterations
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                if has_val:
                    print(
                        f"Epoch {epoch}/{self.epochs}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
                        f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
                else:
                    print(
                        f"Epoch {epoch}/{self.epochs}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")
        return self

    def get_metrics(self):
        """Return the history of losses and accuracies during training."""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

    def get_best_parameters(self):
        """Return the best weights and bias after training."""
        if self.best_weights is None or self.best_bias is None:
            raise ValueError("Model hasn't been trained with early stopping.")
        return self.best_weights, self.best_bias
