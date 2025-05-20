import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder



# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path:str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
    except Exception as e:
        logger.error("Unexpected Error during loading data from %s", e)
        raise
    
def converting_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric and change data types"""
    try:
        logger.debug("Converting columns process start")
        le = LabelEncoder()

        # Convert person_age to int and filter
        df["person_age"] = df["person_age"].astype(int)
        df = df[df['person_age'] <= 100].copy()

        # Convert and clean person_income
        df['person_income'] = df['person_income'].astype(int)
        df = df[df['person_income'] < 1500000].copy()
        df['person_income'] = df['person_income'] // 100 * 100

        # Process loan amount
        df['loan_amnt'] = df['loan_amnt'].astype(int)
        df['loan_amnt'] = df['loan_amnt'] // 100 * 100

        # Process other numeric columns
        df['loan_int_rate'] = df['loan_int_rate'].astype(int)
        df['loan_percent_income'] = df['loan_percent_income'].astype(int) * 100
        df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype(int)

        # Encode categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])

        logger.debug("Converting columns process completed successfully")
        return df

    except Exception as e:
        logger.error("Error during converting columns to numeric: %s", e)
        raise
    
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    
def main():
    try:
        train_data = load_data('./data/interim/train_preprocessed.csv')
        test_data = load_data('./data/interim/test_preprocessed.csv')
        
        train_data = converting_cols(train_data)
        test_data = converting_cols(test_data)
        
        save_data(train_data, os.path.join("./data","Processed","train_data.csv"))
        save_data(train_data, os.path.join("./data","Processed","test_data.csv"))
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main()   
    
    