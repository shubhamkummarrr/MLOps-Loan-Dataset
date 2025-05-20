import os 
import pandas as pd
import numpy as np 
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger for model building
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Create console handler for logging to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Create file handler for logging to a file
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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

def load_model(file_path: str):
    '''Load the trained model from file'''
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug("Model loaded from: %s", file_path)
        return model
    except FileNotFoundError as e:
        logger.debug("File not fount: %s", e)
    except Exception as e:
        logger.debug("Unexpected Error occurred while loading the model: %s", e)
        raise 
    
def load_data(file_path: str) -> pd.DataFrame:
    """Load Data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from: %s", file_path)
        return data
    except FileNotFoundError as e:
        logger.debug("File not found: %s", e)
    except pd.errors.ParserError as e:
        logger.debug("Failed to parse the CSV file: %s", e)
    except Exception as e:
        raise e
    
def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = model.predict(x_test)
    
        Accuracy = accuracy_score(y_test, y_pred)
        Precision = precision_score(y_test, y_pred)
        Recall = recall_score(y_test, y_pred)
        F1_Score = f1_score(y_test, y_pred)
        
        metrics = {
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1_Score': F1_Score
        }
        
        logger.debug("Model evaluation metrics calculated")
        return metrics
    except Exception as e:
        logger.debug("Error during model Evaluation: %s", e)
        raise 
    
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise
    
def main():
    try:
        params = load_params(params_path='params.yaml')
        model = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_data.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))
            live.log_metric('F1_Score', f1_score(y_test, y_test))

            live.log_params(params)
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
