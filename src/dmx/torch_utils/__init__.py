__all__ = ["detect_device"]
import torch

def detect_device() -> str:
    """Detect device available for torch"""
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"
