import numpy as np

def mise(y_true, y_pred, root=False):
    if not(root):
        return np.mean((y_true - y_pred)**2)
    else:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))