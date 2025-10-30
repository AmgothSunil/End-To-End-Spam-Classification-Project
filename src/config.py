import yaml
import logging

logger = logging.getLogger(__name__)

def load_params(params_path: str) -> dict:
    """Load parameters from YAML file."""
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise