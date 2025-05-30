from pathlib import Path
from model.logistic_regression import LogisticRegression
from utils.data_processing import scale_numerical_features
from utils.metrics import calculate_metrics
from utils.visualization import plot_learning_curve, plot_confusion_matrix
from utils.model_utils import save_checkpoint, setup_logging


class LogisticRegressionTrainer:
    def __init__(self, config):
        self.config = config
        self.model = LogisticRegression(**config.get('params', {}))

    def train(self, X_train, y_train, X_val, y_val, output_dir: Path):
        # Setup logging
        logger = setup_logging(output_dir / 'training_log.txt')
        logger.info(
            f"Training Logistic Regression with parameters: {self.config.get('params', {})}")

        # Scale numerical features
        X_train_scaled, X_val_scaled, scaler = scale_numerical_features(
            X_train, X_val, ['AGE'])
        save_checkpoint(scaler, output_dir, 'scaler')

        # Train model
        self.model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            early_stopping=True,
            patience=10
        )

        # Get training history
        metrics = self.model.get_training_history()
        train_losses = metrics["train_losses"]
        train_accuracies = metrics["train_accuracies"]
        val_losses = metrics["val_losses"]
        val_accuracies = metrics["val_accuracies"]

        # Log training progress
        for i, (loss, acc) in enumerate(zip(train_losses, train_accuracies)):
            logger.info(
                f"Epoch {i + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        # Plot learning curves
        plot_learning_curve(train_losses, val_losses,
                            train_accuracies, val_accuracies, output_dir)

        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        metrics = calculate_metrics(y_val, y_pred)

        # Log validation results
        logger.info(f"Validation results:\n"
                    f"Accuracy: {metrics['accuracy']:.4f}\n"
                    f"Precision: {metrics['precision']:.4f}\n"
                    f"Recall: {metrics['recall']:.4f}\n"
                    f"F1 Score: {metrics['f1']:.4f}\n"
                    f"Classification Report:\n{metrics['classification_report']}")

        # Plot confusion matrix
        plot_confusion_matrix(y_val, y_pred, 'Logistic Regression', output_dir)

        # Save model
        save_checkpoint(self.model, output_dir, 'model')

        return self.model
