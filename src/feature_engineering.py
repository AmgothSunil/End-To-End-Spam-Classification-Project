import os
import logging
import numpy as np
import pandas as pd
from config import load_params
from gensim.models import Word2Vec

# Ensure log directory already exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("Feature Engineering")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#load params
params = load_params("params.yaml")


def get_document_vectors(texts: pd.Series, model: Word2Vec) -> np.ndarray:
    """Convert text documents into average word vectors.
    
    Args:
        texts: Series of preprocessed text documents
        model: Trained Word2Vec model
        
    Returns:
        numpy array of shape (n_documents, vector_size)
    """
    
    try:
        vectors = []
        
        for text in texts:
            if not isinstance(text, str):
                text = ""
            
            words = text.split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            
            if len(word_vectors) > 0:
                avg_vectors = np.mean(word_vectors, axis=0)
            else:
                avg_vectors = np.zeros(model.vector_size)
            
            vectors.append(avg_vectors)
        
        logger.debug("Text documents successfully converted to avg word vectors")
        return np.array(vectors)
    
    except Exception as e:
        logger.error("Unexpected error occurred while converting avg word vectors: %s", e)
        raise


def apply_word2vec(text_column: str, train_data: pd.DataFrame, 
                   test_data: pd.DataFrame) -> tuple:
    """Convert text data into vectors using Word2Vec model from gensim.
    
    Args:
        text_column: Name of text column
        train_data: Training dataframe
        test_data: Test dataframe
        
    Returns:
        Tuple of (train_vectors, test_vectors, word2vec_model)
    """
    
    try:
        feature_eng_params = params['feature_engineer_params']
        word2vec_params = feature_eng_params['word2vec_params']
        
        logger.info("Training started for Word2Vec model on training data")
        
        # Tokenize sentences from preprocessed text
        train_sentences = [text.split() for text in train_data[text_column] 
                          if isinstance(text, str) and len(text.strip()) > 0]
        
        logger.info("Training Word2Vec model with %d sentences...", len(train_sentences))
        
        word2vec_model = Word2Vec(
            sentences=train_sentences,
            vector_size=word2vec_params['vector_size'],
            min_count=word2vec_params['min_count'],
            window=word2vec_params['window'],
            workers=word2vec_params['workers'],
            sg=word2vec_params['sg'],
            epochs=word2vec_params['epochs']
        )
        
        logger.debug("Training completed. Vocabulary size: %d", len(word2vec_model.wv))
        
        # Transform train data into vectors
        logger.info("Transforming train data into vectors...")
        train_vectors = get_document_vectors(train_data[text_column], word2vec_model)
        logger.debug("Train vectors shape: %s", train_vectors.shape)
        
        # Transform test data into vectors
        logger.info("Transforming test data into vectors...")
        test_vectors = get_document_vectors(test_data[text_column], word2vec_model)
        logger.debug("Test vectors shape: %s", test_vectors.shape)
        
        return train_vectors, test_vectors, word2vec_model
    
    except Exception as e:
        logger.error("Unexpected error occurred while applying Word2Vec: %s", e)
        raise


def save_vectors(train_vectors: np.ndarray, test_vectors: np.ndarray,
                train_labels: pd.Series, test_labels: pd.Series,
                word2vec_model: Word2Vec) -> None:
    """Save transformed vectors and Word2Vec model."""
    
    try:
        feature_eng_params = params["feature_engineer_params"]
        
        x_train_path = feature_eng_params['x_train_data_path']
        x_test_path = feature_eng_params['x_test_data_path']
        y_train_path = feature_eng_params['y_train_data_path']
        y_test_path = feature_eng_params['y_test_data_path']
        model_path = feature_eng_params['word2vec_model_path']
        
        logger.info("Saving word vectors and model...")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(x_train_path), exist_ok=True)
        os.makedirs(os.path.dirname(x_test_path), exist_ok=True)  
        os.makedirs(os.path.dirname(y_train_path), exist_ok=True)
        os.makedirs(os.path.dirname(y_test_path), exist_ok=True) 
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.debug("Directories created successfully")
        
        # Save feature vectors
        pd.DataFrame(train_vectors).to_csv(x_train_path, index=False, header=False)
        pd.DataFrame(test_vectors).to_csv(x_test_path, index=False, header=False)
        logger.info(f"shapes: train vectors shape {train_vectors.shape} : test vectors shape {test_vectors.shape}")
        logger.debug("Feature vectors saved")
        
        # Save labels
        train_labels.to_csv(y_train_path, index=False, header=False)
        test_labels.to_csv(y_test_path, index=False, header=False)
        logger.info(f"shapes: train labels shape {train_labels.shape} : test labels shape {test_labels.shape}")

        logger.debug("Labels saved")
        
        # Save Word2Vec model
        word2vec_model.save(model_path)
        logger.debug("Word2Vec model saved successfully")
        
        logger.info("All vectors and model saved successfully")
    
    except Exception as e:
        logger.error("Unexpected error occurred while saving vectors and model: %s", e)
        raise


def main():
    """Main feature engineering pipeline."""
    
    try:
        feature_eng_params = params['feature_engineer_params']
        
        text_column = feature_eng_params['text_column']
        target_column = feature_eng_params['target_column']
        train_data_path = feature_eng_params['train_data_path']
        test_data_path = feature_eng_params['test_data_path']
        
        logger.info("Started feature engineering pipeline")
        
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        logger.info("Data loaded. Train shape: %s, Test shape: %s", 
                   train_data.shape, test_data.shape)
        
        # Apply Word2Vec
        logger.info("Applying Word2Vec transformation...")
        train_vectors, test_vectors, word2vec_model = apply_word2vec(
            text_column, train_data, test_data
        )
        
        # Extract labels from original DataFrames (NOT from vectors!)
        train_labels = train_data[target_column]
        test_labels = test_data[target_column]
        logger.debug("Features and labels extracted successfully")
        
        # Save vectors and model
        save_vectors(train_vectors, test_vectors, 
                    train_labels, test_labels, 
                    word2vec_model)
        
        logger.info("Feature engineering pipeline completed successfully!")
    
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    
    except KeyError as e:
        logger.error("Missing configuration key: %s", e)
        raise
    
    except Exception as e:
        logger.error("Feature engineering pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()