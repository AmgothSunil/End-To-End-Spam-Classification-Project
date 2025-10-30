import os
import pandas as pd
import logging
from config import load_params
from sklearn.model_selection import train_test_split

## Create log directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

## Logging configuration
logger = logging.getLogger("Data Ingestion")
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

#load params from params.yaml file
params = load_params('params.yaml')

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    
    try:
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, encoding='latin-1')
        logger.debug("Data loaded successfully. Shape: %s", df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error while loading the data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data"""
    
    try:
        logger.info("Started data preprocessing...")
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
        df = df.rename(columns={'v1': "label", 'v2': 'text'})
        df = df.drop_duplicates(keep='first')
        logger.debug("Data preprocessing completed. Shape: %s", df.shape)
        return df
    
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise



def save_data(dataframe: pd.DataFrame, data_path: str) -> None:
    """Save train and test datasets."""
    
    try:
        ingestion_params = params['data_ingestion']

        test_size = ingestion_params['test_size']
        random_state= ingestion_params['random_state']
        
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data, test_data = train_test_split(
            dataframe, test_size=test_size, random_state=random_state
        )
        
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug("Train and test data saved to %s", raw_data_path)
    
    except Exception as e:
        logger.error("Unexpected error while saving the data: %s", e)
        raise


def main():
    try:
        ingestion_params = params['data_ingestion']

        souce_data_path = ingestion_params['source_data_path']
        save_dir = ingestion_params['save_dir']

        df = load_data(souce_data_path)
        final_df = preprocess_data(df)
        save_data(final_df, save_dir)
        logger.info("Data ingestion process completed successfully")
    
    except Exception as e:
        logger.error("Failed to complete data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()