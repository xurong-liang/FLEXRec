import torch
import numpy as np
import random

def print_text(text, fp=None):
    """
    Print text to stdout and optionally to a file.
    
    Args:
        text (str): Text to print
        fp (file object, optional): File handle to print to. If None, only print to stdout.
    """
    # Print to stdout
    print(text)
    
    # If file pointer provided, print to file
    if fp is not None:
        print(text, file=fp)
        # Ensure content is written immediately
        fp.flush()
        
        
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
