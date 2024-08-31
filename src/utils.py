import yaml
import logging
import os
import torch

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_file='app.log'):
    log_path = os.path.join(get_project_root(), log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    try:
        full_path = os.path.join(get_project_root(), config_path)
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {full_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {full_path}: {str(e)}")
        raise

def save_model(model, path):
    try:
        full_path = os.path.join(get_project_root(), path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(model.state_dict(), full_path)
        logging.info(f"Model saved to {full_path}")
    except Exception as e:
        logging.error(f"Error saving model to {full_path}: {str(e)}")
        raise

def load_model(model_class, path, device, **kwargs):
    try:
        full_path = os.path.join(get_project_root(), path)
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(full_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"Model loaded from {full_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {full_path}: {str(e)}")
        raise