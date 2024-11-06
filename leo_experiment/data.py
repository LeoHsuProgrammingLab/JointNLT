import numpy as np
from huggingface_hub import snapshot_download

class Normalizer:
    def __init__(self):
        pass
    
    def min_max_normalize(self, data):
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

if __name__ == "__main__":
    main()