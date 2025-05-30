import numpy as np

class MLP:
    def __init__(self, input_size=None, hidden_sizes=[64, 32], output_size=1, threshold=0.5,
                 learning_rate=0.001, activation='relu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.threshold = threshold
        
        # Internal attributes
        self.weights = []
        self.biases = []
        self.best_weights = None
        self.best_biases = None
        self.initialized = False
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _initialize_parameters(self, input_size):
        """Initialize weights and biases for all layers"""
        self.input_size = input_size
        
        # Layer dimensions
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        self.weights = []
        self.biases = []
        
        # Weight and bias initialization
        for i in range(len(layer_sizes) - 1):
            if self.activation == 'relu':
                # He initialization for ReLU
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
                
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.initialized = True
    
    def activation_function(self, z):
        """Activation function"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def activation_derivative(self, z):
        """Derivative of the activation function"""
        if self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
    
    def forward(self, X):
        """Forward pass through the MLP"""
        if not self.initialized:
            self._initialize_parameters(X.shape[1])
            
        activations = [X]
        zs = []
        
        a = X
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(a, w) + b
            a = self.activation_function(z)
            
            zs.append(z)
            activations.append(a)
        
        # Output layer always uses sigmoid (for binary classification)
        w = self.weights[-1]
        b = self.biases[-1]
        z = np.dot(a, w) + b
        a = 1 / (1 + np.exp(-z))
        
        zs.append(z)
        activations.append(a)
        
        return activations, zs
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_accuracy(self, y_true, y_pred_proba):
        """Compute binary classification accuracy"""
        y_pred = (y_pred_proba > self.threshold).astype(int)
        return np.mean(y_pred == y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, early_stopping=False, patience=10):
        """Train the MLP model"""
        if not self.initialized:
            self._initialize_parameters(X_train.shape[1])
        
        # Convert pandas to numpy if necessary
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if X_val is not None and hasattr(X_val, 'values'):
            X_val = X_val.values
        if y_val is not None and hasattr(y_val, 'values'):
            y_val = y_val.values
    
        # Ensure targets are 2D
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if X_val is not None and y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        
        best_val_loss = float('inf')
        no_improvement = 0
        
        # Reset history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, zs = self.forward(X_train)
            y_pred = activations[-1]
            
            # Loss and accuracy
            train_loss = self.compute_loss(y_train, y_pred)
            train_accuracy = self.compute_accuracy(y_train, y_pred)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward(X_val)
                val_pred = val_activations[-1]
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.compute_accuracy(y_val, val_pred)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_weights = [w.copy() for w in self.weights]
                        self.best_biases = [b.copy() for b in self.biases]
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    if no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        if self.best_weights is not None:
                            self.weights = self.best_weights
                            self.biases = self.best_biases
                        break
            
            # Backpropagation
            dW = [np.zeros_like(w) for w in self.weights]
            db = [np.zeros_like(b) for b in self.biases]
            
            delta = y_pred - y_train
            m = y_train.shape[0]
            
            dW[-1] = (1/m) * np.dot(activations[-2].T, delta)
            db[-1] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            for l in range(len(self.weights)-2, -1, -1):
                delta = np.dot(delta, self.weights[l+1].T) * self.activation_derivative(zs[l])
                dW[l] = (1/m) * np.dot(activations[l].T, delta)
                db[l] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            # Update parameters
            for l in range(len(self.weights)):
                self.weights[l] -= self.learning_rate * dW[l]
                self.biases[l] -= self.learning_rate * db[l]
            
            # Progress log
            if (epoch+1) % 100 == 0:
                status = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
                if X_val is not None and y_val is not None:
                    status += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                print(status)
        
        # Restore best weights if early stopping was used
        if early_stopping and self.best_weights is not None:
            self.weights = self.best_weights
            self.biases = self.best_biases
        
        return self
    
    def predict_proba(self, X):
        """Predict probability output"""
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict(self, X):
        """Predict class labels (0 or 1)"""
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def get_metrics(self):
        """Return training/validation history"""
        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies
        }
