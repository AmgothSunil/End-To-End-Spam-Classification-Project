import os
import logging
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from config import load_params

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Ensure the log directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging Initialization
logger = logging.getLogger("Data Preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  

file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(file_path)  
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize stemmer and stopwords 
snow_stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english")) 

## load params
params = load_params('params.yaml')

#load label encoder
label_encoder = LabelEncoder()

def transform_text(text: str) -> str:
    """Transform the input text into tokens by removing unnecessary text, 
    lowering the text, removing stopwords and applying stemming"""
    
    try:
        # return empty string if NOT a string
        if not isinstance(text, str):
            logger.warning("Non-string input received: %s", type(text))
            return ""
        
        # remove special characters, keep alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = word_tokenize(text)
        
        # Filter stopwords and apply stemming
        tokens = [snow_stemmer.stem(token) for token in text if token not in stop_words]
        
        return " ".join(tokens)
    
    except Exception as e:
        logger.error("Error occurred while transforming the text: %s", e)
        return ""


def preprocess_df(df: pd.DataFrame, text_column: str , target_column: str ) -> pd.DataFrame:
    """Processing the data by encoding the target column and applying transformation to text column"""
        
    try:

        preprocessing_params = params['data_preprocessing']

        text_column = preprocessing_params['text_column']
        target_column = preprocessing_params['target_column']

        logger.info("Started processing the data...")
        
        # Encode target column
        df[target_column] = label_encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")
        
        # Transform text column
        logger.info("Transforming text column (this may take a while)...")
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        
        return df
    
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


def main():
    """Load the data, process data and save the processed data"""
    
    try:

        
        preprocessing_params = params['data_preprocessing']
        
        text_column = preprocessing_params['text_column']
        target_column = preprocessing_params['target_column']
        train_data_path = preprocessing_params['train_data_path']
        test_data_path = preprocessing_params['test_data_path']
        output_path = preprocessing_params['output_path']

        logger.info("Main function started...")

        # Load the data
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.info("Data loaded. Train shape: %s, Test shape: %s", 
                   train_data.shape, test_data.shape)
        
        # Transform the data
        train_data_processed = preprocess_df(train_data, text_column, target_column)
        test_data_processed = preprocess_df(test_data, text_column, target_column)
        
        # Save processed data
        os.makedirs(output_path, exist_ok=True)
        
        train_data_processed.to_csv(os.path.join(output_path, 'train_processed.csv'), index=False)
        test_data_processed.to_csv(os.path.join(output_path, 'test_processed.csv'), index=False)
        logger.info("Processed data stored successfully at %s", output_path)
        logger.debug("Data Preprocessing pipeline processed successfully")
    
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
        raise
    
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        raise


if __name__ == "__main__":
    main()