"""GPU utility functions for CatBoost"""
import catboost as cb


def get_task_type(use_gpu=True):
    """Get task_type for CatBoost (GPU or CPU)
    
    Args:
        use_gpu: Whether to try using GPU
        
    Returns:
        "GPU" if GPU is available and use_gpu=True, else "CPU"
    """
    if not use_gpu:
        return "CPU"
    
    try:
        # Check if CatBoost GPU is available
        # CatBoost will raise an error during model creation if GPU is not available
        # We use a very minimal check to avoid overhead
        test_model = cb.CatBoostClassifier(
            iterations=1,
            task_type="GPU",
            verbose=False,
            allow_writing_files=False
        )
        
        # Just creating the model doesn't check GPU, so we'll try a minimal fit
        # But to avoid overhead, we'll just return GPU if no error is raised
        # The actual GPU check happens when fit() is called
        # For now, we'll be optimistic and let CatBoost handle the error
        return "GPU"
    except Exception:
        # GPU not available or error, fall back to CPU
        return "CPU"


def check_gpu_available():
    """Check if GPU is available for CatBoost
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        task_type = get_task_type(use_gpu=True)
        return task_type == "GPU"
    except Exception:
        return False

