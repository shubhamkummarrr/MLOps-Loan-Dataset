import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensure that log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url:str) -> pd.DataFrame:
    """Load data from csv file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded from {data_url} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"failed to parse the CSV file: {e}")
        raise
        
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df = df.drop_duplicates()
        df = df.dropna()
        logger.debug("Data preprocess completed")
        return df
    except Exception as e:
        logger.error("Unexpected error during preprocessing:%s",e)
        raise
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str) ->None:
    '''Save the train and test Datasets'''
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug("Train and Test datasets saved successfully to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occured while saving the data %s", e)
        raise
    
    