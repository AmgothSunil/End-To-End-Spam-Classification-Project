import os
import json
import logging
import pickle
import numpy as np
import pandas as pd

from config import load_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger initialization
logger = logging.getLogger("Model Evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Load params data
params = load_params("params.yaml")


def load_data(x_test_path: str, y_test_path: str, 
              model_path: str) -> tuple:
    """
    Load testing data from CSV files and trained model from pickle file.
    
    Args:
        x_test_path: Path to retrieve x_test data from CSV file
        y_test_path: Path to retrieve y_test data from CSV file
        model_path: Path to retrieve trained model from pickle file
    
    Returns:
        Tuple of (x_test_data, y_test_data, model)
        
    Raises:
        FileNotFoundError: If any of the required files don't exist
        pd.errors.EmptyDataError: If CSV files are empty
        pickle.UnpicklingError: If model file is corrupted
    """
    
    try:
        logger.info("Loading test data and trained model...")
        
        # Load test features
        x_test_data = pd.read_csv(x_test_path, header=None).values
        logger.debug("X_test loaded. Shape: %s", x_test_data.shape)
        
        # Load test labels 
        y_test_data = pd.read_csv(y_test_path, header=None).values.ravel() # Flatten to 1D array
        logger.debug("y_test loaded. Shape: %s", y_test_data.shape)
        
        # Validate data shapes match
        if len(x_test_data) != len(y_test_data):
            raise ValueError(
                f"X_test and y_test lengths don't match: "
                f"{len(x_test_data)} vs {len(y_test_data)}"
            )
        
        # Load trained model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded from %s", model_path)
        
        # Validate model type
        if not hasattr(model, 'predict'):
            raise TypeError(f"Loaded object is not a valid model: {type(model)}")
        
        logger.info("Data and model loaded successfully")
        
        return x_test_data, y_test_data, model
    
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    
    except pd.errors.EmptyDataError as e:
        logger.error("CSV file is empty: %s", e)
        raise
    
    except pickle.UnpicklingError as e:
        logger.error("Failed to load model (corrupted file): %s", e)
        raise
    
    except ValueError as e:
        logger.error("Data validation error: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def evaluate_model(x_test_data: np.ndarray, 
                   y_test_data: np.ndarray,
                   model: RandomForestClassifier) -> dict:
    """
    Evaluate trained model using test datasets with various metrics.
    
    Args:
        x_test_data: Testing feature DataFrame
        y_test_data: Testing labels Series
        model: Trained RandomForestClassifier model
    
    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - roc_auc: ROC AUC score
        
    Raises:
        ValueError: If predictions fail or data is invalid
    """
    
    try:
        logger.info("Evaluating model on test dataset...")
        logger.debug("Test set size: %d samples", len(x_test_data))
        
        # Make predictions
        y_pred_data = model.predict(x_test_data)
        
        # Get prediction probabilities for ROC AUC
        y_pred_prob = model.predict_proba(x_test_data)[:, 1]
        
        # Calculate metrics
        acc_score = accuracy_score(y_test_data, y_pred_data)
        pre_score = precision_score(y_test_data, y_pred_data, zero_division=0)
        rec_score = recall_score(y_test_data, y_pred_data, zero_division=0)
        f1_scre = f1_score(y_test_data, y_pred_data, zero_division=0)
        roc_auc_scre = roc_auc_score(y_test_data, y_pred_prob)
        
        metrics_dict = {
            "accuracy": float(acc_score),
            "precision": float(pre_score),
            "recall": float(rec_score),
            "f1_score": float(f1_scre),
            "roc_auc": float(roc_auc_scre)
        }
        
        logger.info("Model Evaluation Results:")
        logger.info("  Accuracy:  %.4f", acc_score)
        logger.info("  Precision: %.4f", pre_score)
        logger.info("  Recall:    %.4f", rec_score)
        logger.info("  F1 Score:  %.4f", f1_scre)
        logger.info("  ROC AUC:   %.4f", roc_auc_scre)
        
        # Log classification report
        report = classification_report(y_test_data, y_pred_data, zero_division=0)
        logger.debug("Classification Report:\n%s", report)
        
        # Log confusion matrix
        cm = confusion_matrix(y_test_data, y_pred_data)
        logger.debug("Confusion Matrix:\n%s", cm)
        
        logger.info("Model evaluation completed successfully")
        
        return metrics_dict
    
    except ValueError as e:
        logger.error("Error during prediction or metric calculation: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, metrics_json_file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        metrics_json_file_path: Path where metrics JSON will be saved
        
    Raises:
        OSError: If directory creation or file writing fails
        TypeError: If metrics contain non-serializable types
    """
    
    try:
        logger.info("Saving evaluation metrics to %s", metrics_json_file_path)
        
        # Create directory if it doesn't exist
        metrics_dir = os.path.dirname(metrics_json_file_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
            logger.debug("Metrics directory ensured: %s", metrics_dir)
        
        # Save metrics as JSON with pretty formatting
        with open(metrics_json_file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        
        # Verify file was created
        file_size = os.path.getsize(metrics_json_file_path)
        logger.info("Metrics saved successfully (size: %d bytes)", file_size)
        
        # Log metrics content
        logger.debug("Saved metrics: %s", json.dumps(metrics, indent=2))
    
    except OSError as e:
        logger.error("File system error while saving metrics: %s", e)
        raise
    
    except TypeError as e:
        logger.error("Metrics contain non-JSON-serializable data: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error while saving metrics: %s", e)
        raise


def main():
    """Main model evaluation pipeline."""
    
    try:
        logger.info("Starting Model Evaluation Pipeline")
        
        # Get evaluation parameters
        model_eval_params = params['model_evaluation']
        
        x_test_path = model_eval_params['x_test_data_path']
        y_test_path = model_eval_params['y_test_data_path']
        model_path = model_eval_params['rf_model_path']
        metrics_path = model_eval_params['evaluation_metric_path']
        
        # Step 1: Load data and model
        logger.info("Step 1/3: Loading test data and trained model...")
        x_test_data, y_test_data, model = load_data(
            x_test_path, y_test_path, model_path
        )
        
        # Step 2: Evaluate model
        logger.info("Step 2/3: Evaluating model performance...")
        evaluation_metrics = evaluate_model(x_test_data, y_test_data, model)
        
        # Step 3: Save evaluation metrics
        logger.info("Step 3/3: Saving evaluation metrics...")
        save_metrics(evaluation_metrics, metrics_path)
        
        logger.info("Model Evaluation Pipeline Completed Successfully!")
        logger.info("Accuracy: %.4f", evaluation_metrics['accuracy'])
    
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
        logger.error("Model evaluation pipeline failed")
        raise
    
    except KeyError as e:
        logger.error("Missing configuration parameter: %s", e)
        logger.error("Model evaluation pipeline failed")
        raise
    
    except ValueError as e:
        logger.error("Invalid data or parameter: %s", e)
        logger.error("Model evaluation pipeline failed")
        raise
    
    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()