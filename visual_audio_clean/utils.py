import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    Ensures that numpy arrays and other numpy data types can be properly serialized to JSON.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif obj is None:
            return None
        return super(NumpyEncoder, self).default(obj)