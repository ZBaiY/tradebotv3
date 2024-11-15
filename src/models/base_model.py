# base_model.py
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
## historical data handler for backtesting
from src.data_handling.historical_data_handler import HistoricalDataHandler
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor
import src.signal_processing.filters as filter
import src.signal_processing.transform as transform
import numpy as np
import pandas as pd


"""
The Model module will forecast 30 data points into the future.
For each model, there is a trusted future length. Which is smaller than 30.
All the model will be trained by using their trusted length's prediction.
"""


class BaseModel:
    def __init__(self, symbol, data_handler, signal_processors, feature_extractor):
        self.symbol = symbol
        self.trusted_future = 0
        self.forcast_length = 30
        self.data_handler = data_handler
        self.signal_processors = signal_processors
        self.feature_extractor = feature_extractor
        self.symbols = self.data_handler.symbols
        self.prediction = []
        

    
    def train(self, data):
        raise NotImplementedError("Train method must be implemented by the subclass")
    
    def predict(self, data):
        raise NotImplementedError("Predict method must be implemented by the subclass")
    
    def preprocess(self, data):
        # General preprocessing that might apply to all models
        raise NotImplementedError("Preprocess method must be implemented by the subclass")
    
    def evaluate(self, data):
        # Evaluation method for all models
        raise NotImplementedError("Evaluate method must be implemented by the subclass")
    

class ForTesting(BaseModel):
    def __init__(self, symbol, data_handler):
        super().__init__(data_handler)
        self.symbol = symbol
        self.trusted_future = 10
        es_filter = filter.ExponentialSmoothingFilter(alpha=0.3)
        rt_trans = transform.LogReturnTransformer()
        filters = [es_filter]
        transformers = [rt_trans]
        self.rts_processor = NonMemSignalProcessor(self.data_handler, 'close',filters, transformers)

    
    def train(self, data):
        return data
    
    def predict(self):
        rts = self.rts_processor.apply_all().bfill().ffill()
        rts_mean = [rts[symbol].mean() for symbol in self.symbols]
        rts_cov = [rts[symbol].cov() for symbol in self.symbols]
        prediction = {[] for symbol in self.symbols}
        rts_pred = {[] for symbol in self.symbols}
        for symbol in self.symbols:
            rts_pred[symbol] = np.random.multivariate_normal(rts_mean[symbol], rts_cov[symbol], self.forcast_length)
            last_price = self.data_handler.get_last_data(symbol)['close']
            prediction[symbol] = last_price * np.exp(rts_pred[symbol].cumsum())

        self.prediction = prediction
    
    def preprocess(self):
        pass
    
    def evaluate(self, data):
        return data