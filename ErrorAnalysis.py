import sklearn.metrics as sk
import numpy as np
import pandas as pd

class ErrorAnalysis:
    
    def __init__(self,series,predict):

        self.criterias = pd.Series()               
        self.criterias.set_value('MAE',sk.mean_absolute_error(series,predict))
        self.criterias.set_value('RMSE', np.sqrt(sk.mean_squared_error(series,predict)))
        self.criterias.set_value('MAPE', self.criterias['MAE'] / np.mean(series) * 100)


