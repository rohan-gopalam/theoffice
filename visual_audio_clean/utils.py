# utils.py
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types. """
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add any other general utility functions here if needed later