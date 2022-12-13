import numpy as np

class DataDescriptor:
    def __init__(self, data):
        self.data = data
    
    def close_minus_mean(self, data):
        data["close_mean_diff"] = np.abs(data["close"] - (
            (data["sma_high"] + data["sma_low"]) / 2
        ))
        return data

    def run(self):
        data = self.data.copy()
        data = self.close_minus_mean(data)
        return data
