"""
Threshold calculation module.
Loads trained linear regression model and returns threshold value.
"""
import json
from pathlib import Path

# Default threshold if model not found
DEFAULT_THRESHOLD = 0.0
MODEL_FILE = Path(__file__).parent / "threshold_model.json"

def get_threshold():
    """
    Get threshold value from trained model.
    
    Returns:
        float: Threshold value for uncertainty comparison
    """
    if not MODEL_FILE.exists():
        # Model not trained yet, return default
        return DEFAULT_THRESHOLD
    
    try:
        with open(MODEL_FILE, 'r') as f:
            model_data = json.load(f)
        return model_data.get('threshold', DEFAULT_THRESHOLD)
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Warning: Error loading threshold model: {e}")
        print(f"Using default threshold: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD

def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information or None if model not found
    """
    if not MODEL_FILE.exists():
        return None
    
    try:
        with open(MODEL_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None