from pathlib import Path
import numpy as np

from model.naive_bayes import NaiveBayes
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix
from utils.model_utils import save_checkpoint, setup_logging
from utils.data_processing import scale_numerical_features


class NaiveBayesTrainer:
    def __init__(self, config):
        self.config = config
        self.model = NaiveBayes(**config.get('params', {}))

    def train(self, X_train, y_train, X_val, y_val, output_dir: Path):
        # Setup logging
        logger = setup_logging(output_dir / 'training_log.txt')
        logger.info(
            f"Training Naive Bayes with parameters: {self.config.get('params', {})}")

        # Scale numerical features
        X_train_scaled, X_val_scaled, scaler = scale_numerical_features(
            X_train, X_val, ['AGE'])
        save_checkpoint(scaler, output_dir, 'scaler')

        X_train_scaled = X_train_scaled.to_numpy()
        X_val_scaled = X_val_scaled.to_numpy()

        # Train model
        self.model.fit(
            X=X_train_scaled,
            y=y_train
        )
        y_pred_train = self.model.predict(X_train_scaled)
        train_metrics = self.model.compute_metrics(y_train, y_pred_train)

        y_val_pred = self.model.predict(X_val_scaled)
        val_metrics = self.model.compute_metrics(y_val, y_val_pred)

        # Log metrics
        logger.info(f"Training metrics:\n"
                    f"Accuracy: {train_metrics['accuracy']:.4f}\n"
                    f"Precision: {train_metrics['precision']:.4f}\n"
                    f"Recall: {train_metrics['recall']:.4f}\n"
                    f"F1 Score: {train_metrics['f1']:.4f}\n"
                    f"Classification Report:\n{train_metrics['classification_report']}\n\n")

        logger.info(f"Validation results:\n"
                    f"Accuracy: {val_metrics['accuracy']:.4f}\n"
                    f"Precision: {val_metrics['precision']:.4f}\n"
                    f"Recall: {val_metrics['recall']:.4f}\n"
                    f"F1 Score: {val_metrics['f1']:.4f}\n"
                    f"Classification Report:\n{val_metrics['classification_report']}")

        # save model
        save_checkpoint(self.model, output_dir, 'NB_model')

        return self.model
