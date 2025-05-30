from pathlib import Path
from model.mlp import MLP
from utils.data_processing import scale_numerical_features
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_learning_curve
from utils.model_utils import save_checkpoint, setup_logging

class MLPTrainer:
    def __init__(self, config):
        self.config = config
        params = config.get('params', {})
        self.epochs = params.get('epochs', 1000)
        
        # Initialize the MLP model with parameters from config
        self.model = MLP(
            input_size=params.get('input_size', None), 
            hidden_sizes=params.get('hidden_sizes', [64, 32]),
            output_size=params.get('output_size', 1), 
            learning_rate=params.get('learning_rate', 0.001),
            activation=params.get('activation', 'relu')
        )

    def train(self, X_train, y_train, X_val, y_val, output_dir: Path):
        # Set up logging to a file
        logger = setup_logging(output_dir / 'training_log.txt')
        logger.info(f"Training MLP with parameters: {self.config.get('params', {})}")

        # Scale numerical features (e.g., 'AGE') and save the scaler
        X_train_scaled, X_val_scaled, scaler = scale_numerical_features(X_train, X_val, ['AGE'])
        save_checkpoint(scaler, output_dir, 'scaler')

        # Log the shape of the training input data
        logger.info(f"Input data shape: {X_train_scaled.shape}")
        
        # Train the model using early stopping to prevent overfitting
        self.model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            epochs=self.epochs, 
            early_stopping=True,
            patience=5
        )
        
        # Retrieve the training and validation metrics
        metrics = self.model.get_metrics()
        train_losses = metrics["train_losses"]
        train_accuracies = metrics["train_accuracies"]
        val_losses = metrics["val_losses"]
        val_accuracies = metrics["val_accuracies"]

        # Log loss and accuracy for each epoch
        for i, (loss, acc) in enumerate(zip(train_losses, train_accuracies)):
            logger.info(f"Epoch {i + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
                
        # Evaluate model performance on validation set
        y_pred = self.model.predict(X_val_scaled)
        eval_metrics = calculate_metrics(y_val, y_pred)
        
        # Log final evaluation metrics
        logger.info(f"Validation results:\n"
                   f"Accuracy: {eval_metrics['accuracy']:.4f}\n"
                   f"Precision: {eval_metrics['precision']:.4f}\n"
                   f"Recall: {eval_metrics['recall']:.4f}\n"
                   f"F1 Score: {eval_metrics['f1']:.4f}\n"
                   f"Classification Report:\n{eval_metrics['classification_report']}")
        
        # Plot and save the confusion matrix
        plot_confusion_matrix(y_val, y_pred, 'MLP', output_dir)
        
        # Plot and save learning curves for loss and accuracy over epochs
        if hasattr(self.model, 'train_losses'):
            plot_learning_curve(
                self.model.train_losses, 
                self.model.val_losses if hasattr(self.model, 'val_losses') else None,
                self.model.train_accuracies if hasattr(self.model, 'train_accuracies') else None,
                self.model.val_accuracies if hasattr(self.model, 'val_accuracies') else None,
                output_dir
            )
        
        # Save the trained model to disk
        save_checkpoint(self.model, output_dir, 'model')
        
        return eval_metrics
