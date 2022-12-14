import numpy as np

"""
TODO:
1. medias de 100 en 100 hasta 1000
2. distancia entre en close y cada media
3. promedio de todas las medias
4. promedio de las distancias
5. diferencia entre el high y 3.
6. diferencia entre el low y 3.
7. desviación estándar de 100 en 100 hasta 1000
8. promedio de las desviaciones estándar
9. incremento de cada media (resta del valor actual y el anterior)
10. incremento de cada desviación (resta del valor actual y el anterior)
"""
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
