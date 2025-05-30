import argparse
from pathlib import Path
from utils.data_processing import load_data
from utils.model_utils import load_config, create_output_dir
from train import LogisticRegressionTrainer, RandomForestTrainer, DecisionTreeTrainer, KNeighborsTrainer, MLPTrainer

# Mapping model names to trainer classes
TRAINER_CLASSES = {
    'LogisticRegression': LogisticRegressionTrainer,
    'RandomForest': RandomForestTrainer,
    'DecisionTree': DecisionTreeTrainer,
    'KNN': KNeighborsTrainer, 
    'MLP': MLPTrainer, 
    # Add other trainers as they are implemented
}


def main(args):
    # Load configuration
    config = load_config(args.config)
    model_type = config["model"]["type"]

    if model_type not in TRAINER_CLASSES:
        raise ValueError(
            f"Model '{model_type}' is not supported. Choose one of: {list(TRAINER_CLASSES.keys())}")

    # Create output directory
    output_dir = create_output_dir(model_type)

    # Load data
    X_train, y_train = load_data(args.train_path)
    X_val, y_val = load_data(args.val_path)

    # Initialize and train model
    TrainerClass = TRAINER_CLASSES[model_type]
    trainer = TrainerClass(config["model"])
    trainer.train(X_train, y_train, X_val, y_val, output_dir)

    print(f"Training completed successfully. Results saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file (YAML format)')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to the training data CSV file')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to the validation data CSV file')
    args = parser.parse_args()

    main(args)
