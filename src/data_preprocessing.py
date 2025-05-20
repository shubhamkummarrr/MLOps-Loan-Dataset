import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_df(df: pd.DataFrame):
    """Preprocessing DataFrame by encoding the data and removing duplicates and unknown values"""
    try:
        logger.debug("Starting preprocessing for DataFrame")
        
        # Applying LabelEnder to all the categorical columns
        le = LabelEncoder()
        df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(le.fit_transform)

        logger.debug("Text preprocessed completed")
        return df
    except KeyError as e:
        logger.error("Something wrong happen with columns: %s", e)
        raise
    except Exception as e:
        logger.error("Error during data_preprocessing: %s",e)
        raise
    
def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded properly")
        
        # Transform the data
        train_preprocessed_data = preprocess_df(train_data)
        test_preprocessed_data = preprocess_df(test_data)
                
        # Store the processed data inside data/interim
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_preprocessed_data.to_csv(os.path.join(data_path, 'train_preprocessed.csv'), index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path, 'test_preprocessed.csv'), index=False)
        
        logger.debug("Processed data saved to %s", data_path)
    
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
