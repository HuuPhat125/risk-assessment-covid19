import yaml
from pathlib import Path
import joblib
import logging

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_dir(model_type, root_dir='./results'):
    """Create a directory for model training outputs."""
    model_dir = Path(root_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in model_dir.iterdir() if d.is_dir()
                and d.name.startswith('run_')]
    indices = [int(d.name.split('_')[1])
               for d in existing if d.name.split('_')[1].isdigit()]
    next_index = max(indices, default=0) + 1
    run_dir = model_dir / f'run_{next_index}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_checkpoint(obj, save_dir, name):
    """Save model or scaler checkpoint"""
    save_path = save_dir / f'{name}.joblib'
    joblib.dump(obj, save_path)

def load_checkpoint(save_dir, name):
    """Load model or scaler checkpoint"""
    load_path = save_dir / f'{name}.joblib'
    return joblib.load(load_path)

def setup_logging(log_path):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
