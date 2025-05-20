# Import necessary libraries
import os
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger for model building
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Create console handler for logging to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Create file handler for logging to a file
log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Set log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse to CSV file: %s", e)
    except FileNotFoundError as e:
        logger.error("file not found: %s", e)
    except Exception as e:
        raise 
    
def initialize_xgb_classifier(params: dict) -> XGBClassifier:
    """
    Initialize the XGBClassifier with the given parameters.
    """
    logger.debug("Initializing XGBClassifier with parameters")
    return XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42,
        eval_metric='logloss'
    )

def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    try:
        model = initialize_xgb_classifier(params)
        # Log the number of training samples
        logger.debug("Model training started with %d samples", x_train.shape[0])

        model.fit(x_train, y_train)

        logger.debug("Model training completed")

        return model
    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise
    
def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')["model_building"]
        train_data = load_data('./data/Processed/train_data.csv')
        
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        model = train_model(x_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        
        save_model(model, model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()
