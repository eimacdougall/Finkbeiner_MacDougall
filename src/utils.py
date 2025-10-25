import os
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def compare_predicted(pred, actual):
    return pred == actual