from pathlib import Path
from model.knn import KNeighbors
from utils.metrics import calculate_metrics
from utils.model_utils import save_checkpoint, setup_logging

class KNeighborsTrainer:
    def __init__(self, config):
        self.config = config
        self.model = KNeighbors(**config.get('params', {}))

    def train(self, X_train, y_train, X_val, y_val, output_dir: Path):
        # Setup log 
        logger = setup_logging(output_dir / 'training_log.txt')
        logger.info(f"K Nearest Neighbors with parameters: {self.config.get('params', {})}")

        # fit 
        self.model.fit(X=X_train, y=y_train)
        y_pred_train = self.model.predict(X_train)
        train_metrics = calculate_metrics(y_pred_train, y_train)

        # validation
        y_pred = self.model.predict(X_val) 
        val_metrics = calculate_metrics(y_val, y_pred)
        
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
        save_checkpoint(self.model, output_dir, 'KNN_model')

        return self.model
