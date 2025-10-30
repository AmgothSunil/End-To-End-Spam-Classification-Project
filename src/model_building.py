import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import load_params

# Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger initialization
logger = logging.getLogger("Model Training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load params data
params = load_params('params.yaml')


def load_data(x_train_path: str, y_train_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data from CSV files.
    
    Args:
        x_train_path: Path to training features CSV file
        y_train_path: Path to training labels CSV file
        
    Returns:
        Tuple of (x_train_data, y_train_data)
    """
    try:
        logger.info("Loading training data from: %s and %s", x_train_path, y_train_path)
        
        x_train_data = pd.read_csv(x_train_path, header=None)
        y_train_data = pd.read_csv(y_train_path, header=None)
        
        logger.debug("Data loaded successfully. X_train shape: %s, y_train shape: %s", 
                    x_train_data.shape, y_train_data.shape)
        
        return x_train_data, y_train_data
    
    except FileNotFoundError as e:
        logger.error("File not found error: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def train_model(x_train_data: pd.DataFrame, y_train_data: pd.DataFrame) -> RandomForestClassifier:
    """Train RandomForestClassifier model.
    
    Args:
        x_train_data: Training features DataFrame
        y_train_data: Training labels DataFrame
        
    Returns:
        Trained RandomForestClassifier model
    """
    try:
        logger.info("Starting model training...")
        model_train_params = params['model_training']
        rf_params = model_train_params['rf_params']
        
        # Convert to numpy arrays 
        x_train = x_train_data.values
        y_train = y_train_data.values.ravel()  # Flatten to 1D array

        if len(x_train) != len(y_train):
            raise ValueError(
                f"x_train and y_train sample counts don't match: "
                f"{len(x_train)} vs {len(y_train)}"
            )

        
        logger.debug("Training data prepared. X shape: %s, y shape: %s", 
                    x_train.shape, y_train.shape)
        
        logger.info("Initializing RandomForestClassifier with params: %s", rf_params)
        
        clf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            criterion=rf_params['criterion'],
            class_weight=rf_params['class_weight'],
            n_jobs=rf_params['n_jobs'],
            random_state=rf_params['random_state']
        )
        
        logger.info("Training model...")
        clf_model.fit(x_train, y_train)
        
        logger.info("Model trained successfully. Feature importance shape: %s", 
                   clf_model.feature_importances_.shape)
        
        return clf_model
    
    except KeyError as e:
        logger.error("Missing configuration key: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occurred while training model: %s", e)
        raise


def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save trained RandomForestClassifier model.
    
    Args:
        model: Trained RandomForestClassifier model
        model_path: Path where model should be saved
    """
    try:
        logger.info("Saving trained model to: %s", model_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.debug("Model saved successfully at: %s", model_path)
    
    except Exception as e:
        logger.error("Unexpected error occurred while saving model: %s", e)
        raise


def main():
    """Main pipeline: load data, train model, and save model."""
    try:
        logger.info("Starting model training pipeline")
        
        model_train_params = params['model_training']
        
        x_train_path = model_train_params['x_train_data_path']
        y_train_path = model_train_params['y_train_data_path']
        model_path = model_train_params['rf_model_path']
        
        # Load data
        logger.info("Step 1: Loading training data...")
        x_train_data, y_train_data = load_data(x_train_path, y_train_path)
        
        # Train model
        logger.info("Step 2: Training Random Forest model...")
        rf_trained_model = train_model(x_train_data, y_train_data)
        
        # Save model
        logger.info("Step 3: Saving trained model...")
        save_model(rf_trained_model, model_path)
        
        logger.info("Model training pipeline completed successfully!")
    
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
        raise
    
    except KeyError as e:
        logger.error("Missing configuration parameter: %s", e)
        raise
    
    except Exception as e:
        logger.error("Model training pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()