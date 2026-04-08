__all__ = ["detect_device"]

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def detect_device() -> str:
    """Detect device available for torch
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
        
    Raises:
        ImportError: If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to use dmx.torch_utils but is not installed.\n"
            "Install with: poetry install --with torch\n"
            "Or: pip install torch"
        )
    
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"
