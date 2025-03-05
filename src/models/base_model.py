# base_model.py
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
## historical data handler for backtesting
from src.data_handling.historical_data_handler import HistoricalDataHandler, SingleSymbolDataHandler, MultiSymbolDataHandler
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor, NonMemSymbolProcessorDataSymbol
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
        self.forecast_length = 30
        self.data_handler = data_handler
        self.signal_processors = signal_processors
        # Remember that the signal processors are a list of signal processors, each having signals from different symbols
        self.feature_extractor = feature_extractor
        # feature extractor contains data from different symbols
        

    
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
    def __init__(self, symbol, data_handler, signal_processors, feature_extractor, model_variant, **params):
        super().__init__(symbol, data_handler, signal_processors, feature_extractor)
        self.symbol = symbol
        self.trusted_future = 10
        self.model_variant = model_variant # for test mode, this is redundant
        self.params = params
        # es_filter = filter.ExponentialSmoothingFilter(alpha=0.3)
        #filters = [es_filter]
        rt_trans = transform.LogReturnTransformer()
        filters = []
        transformers = [rt_trans]
        if self.data_handler.__class__ == SingleSymbolDataHandler:
            self.rts_processor =  NonMemSymbolProcessorDataSymbol(self.symbol, self.data_handler, 'close', filters, transformers)
        else: self.rts_processor = NonMemSymbolProcessor(self.symbol, self.data_handler, 'close', filters, transformers)
        # print("Test model is used.")
    
    def train(self, data):
        return data
    
    def predict(self, **kwargs):
        rts = self.rts_processor.apply_all().bfill().ffill()
        rts_mean = rts.mean()
        rts_var = rts.var()
        prediction = []
        rts_pred = []
        
        rts_pred = np.random.normal(rts_mean, np.sqrt(rts_var), self.forecast_length)
        last_price = self.data_handler.get_last_data(self.symbol)['close']
        prediction = last_price * np.exp(rts_pred.cumsum())

        return prediction

        
    
    def preprocess(self):
        pass
    
    def evaluate(self, data):
        return data

class MACDModel(BaseModel):
    def __init__(self, symbol, data_handler, signal_processors, feature_extractor, model_variant, **params):
        super().__init__(symbol, data_handler, signal_processors, feature_extractor)
        self.symbol = symbol
        self.trusted_future = 10
        self.model_variant = model_variant
        self.params = params
        # es_filter = filter.ExponentialSmoothingFilter(alpha=0.3)
        #filters = [es_filter]
        filters = []
        transformers = []
        if self.data_handler.__class__ == SingleSymbolDataHandler:
            self.rts_processor =  NonMemSymbolProcessorDataSymbol(self.symbol, self.data_handler, 'close', filters, transformers)
        else: self.rts_processor = NonMemSymbolProcessor(self.symbol, self.data_handler, 'close', filters, transformers)

    def train(self, data):
        return data
    
    def predict(self, **kwarg):
        last_price = self.data_handler.get_last_data(self.symbol)['close']
        prediction = [last_price] * self.forecast_length
        
        if type(self.feature_extractor.indicators) == dict:
            macds = self.feature_extractor.indicators[self.symbol][['macd', 'macd_signal', 'macd_diff']].tail(5)
        elif type(self.feature_extractor.indicators) == pd.DataFrame:
            macds = self.feature_extractor.indicators[['macd', 'macd_signal', 'macd_diff']].tail(5)
        last_diff = macds['macd_diff'].iloc[-1]
        prev_diff = macds['macd_diff'].iloc[-2]
        if last_diff and prev_diff:
            if prev_diff < 0 and last_diff > 0:
            #"BUY - MACD Histogram turned positive"
                for i in range(1, self.forecast_length):
                    prediction[i] = prediction[i-1] * (1 + 0.05)

            elif prev_diff > 0 and last_diff < 0:
            #   "SELL - MACD Histogram turned negative"
                for i in range(1, self.forecast_length):
                    prediction[i] = prediction[i-1] * (1 - 0.05)
            
        return prediction
    
    def preprocess(self):
        pass
    
    def evaluate(self, data):
        return data

class MACDwADX(BaseModel):
    def __init__(self, symbol, data_handler, signal_processors, feature_extractor, model_variant, **params):
        super().__init__(symbol, data_handler, signal_processors, feature_extractor)
        self.symbol = symbol
        self.trusted_future = 10
        self.model_variant = model_variant
        self.params = params
        # Set the thresholds as attributes (defaults: macd_threshold=0.0, adx_threshold=45)
        self.macd_threshold = params.get("macd_threshold", 0.0)
        self.adx_threshold = params.get("adx_threshold", 45)
        # Example: you can add more parameters as needed.
        
        filters = []
        transformers = []
        if self.data_handler.__class__ == SingleSymbolDataHandler:
            self.rts_processor = NonMemSymbolProcessorDataSymbol(self.symbol, self.data_handler, 'close', filters, transformers)
        else:
            self.rts_processor = NonMemSymbolProcessor(self.symbol, self.data_handler, 'close', filters, transformers)
        
        # Define forecast_length attribute (e.g., based on trusted_future or parameters)
        self.forecast_length = self.trusted_future

    def train(self, data):
        return data

    def predict(self, **kwarg):
        # Get the latest close price from the data handler.
        last_price = self.data_handler.get_last_data(self.symbol)['close']
        # Initialize the prediction as a flat forecast.
        prediction = [last_price] * self.forecast_length

        # Retrieve indicators from the feature_extractor.
        if isinstance(self.feature_extractor.indicators, dict):
            df = self.feature_extractor.indicators[self.symbol]
        elif isinstance(self.feature_extractor.indicators, pd.DataFrame):
            df = self.feature_extractor.indicators
        else:
            raise ValueError("Unsupported format for indicators.")

        # Extract the MACD-related columns (assume they exist).
        macds = df[['macd', 'macd_signal', 'macd_diff']].tail(5)
        last_diff = macds['macd_diff'].iloc[-1]
        prev_diff = macds['macd_diff'].iloc[-2]
        
        # Retrieve the latest ADX value.
        current_adx = df['adx'].iloc[-1]

        # Only activate MACD signals if ADX is above the threshold.
        if current_adx >= self.adx_threshold:
            # Check for a bullish MACD crossover: MACD histogram turning positive.
            if prev_diff < self.macd_threshold and last_diff > self.macd_threshold:
                # "BUY - MACD Histogram turned positive"
                for i in range(1, self.forecast_length):
                    prediction[i] = prediction[i-1] * (1 + 0.05)
            # Check for a bearish MACD crossover: MACD histogram turning negative.
            elif prev_diff > self.macd_threshold and last_diff < self.macd_threshold:
                # "SELL - MACD Histogram turned negative"
                for i in range(1, self.forecast_length):
                    prediction[i] = prediction[i-1] * (1 - 0.05)
        # Else, if ADX is below the threshold, no trade signal is generated.
        # The prediction remains flat at the last price.

        return prediction
    
    def preprocess(self):
        pass
    
    def evaluate(self, data):
        return data

class RSIwADX(BaseModel):
    def __init__(self, symbol, data_handler, signal_processors, feature_extractor, model_variant, **params):
        super().__init__(symbol, data_handler, signal_processors, feature_extractor)
        self.symbol = symbol
        self.trusted_future = 10  # forecast horizon
        self.model_variant = model_variant
        self.params = params

        # Set strategy parameters (with defaults)
        self.adx_threshold = params.get("adx_threshold", 20)
        self.variance_factor = params.get("variance_factor", 1.8)
        self.mae = params.get("mae", 0)  # margin adjustment error
        self.rsi_window = params.get("rsi_var_window", 23)  # window for calculating current RSI statistics

        # Set forecast_length attribute (can be adjusted as needed)
        self.forecast_length = self.trusted_future

        filters = []
        transformers = []
        if self.data_handler.__class__ == SingleSymbolDataHandler:
            self.rts_processor = NonMemSymbolProcessorDataSymbol(self.symbol, self.data_handler, 'close', filters, transformers)
        else:
            self.rts_processor = NonMemSymbolProcessor(self.symbol, self.data_handler, 'close', filters, transformers)

    def train(self, data):
        return data

    def predict(self, **kwargs):
        # Get the last close price as a base value for the forecast.
        last_data = self.data_handler.get_last_data(self.symbol)
        last_price = last_data['close']
        forecast = [last_price] * self.forecast_length

        # Retrieve the indicator DataFrame from the feature extractor.
        if isinstance(self.feature_extractor.indicators, dict):
            df = self.feature_extractor.indicators[self.symbol]
        elif isinstance(self.feature_extractor.indicators, pd.DataFrame):
            df = self.feature_extractor.indicators
        else:
            raise ValueError("Unsupported format for indicators.")

        # Calculate current RSI mean and standard deviation using only the last rsi_window values.
        recent_rsi = df['rsi'].tail(self.rsi_window)
        current_rsi_mean = recent_rsi.mean()
        current_rsi_std = recent_rsi.std()

        # Get current indicator values from the latest row.
        current_rsi = df['rsi'].iloc[-1]
        current_adx = df['adx'].iloc[-1]

        # Calculate dynamic RSI thresholds using the current rolling statistics.
        dynamic_upper = current_rsi_mean + self.variance_factor * current_rsi_std
        dynamic_lower = current_rsi_mean - self.variance_factor * current_rsi_std
        
        # Apply the dynamic RSI strategy only if the market is sideways.
        if current_adx < self.adx_threshold:
            if current_rsi <= dynamic_lower + self.mae:
                # Signal for upward movement (BUY) – forecast a 5% increase per period.
                for i in range(1, self.forecast_length):
                    forecast[i] = forecast[i - 1] * (1 + 0.05)
            elif current_rsi >= dynamic_upper - self.mae:
                # Signal for downward movement (SELL) – forecast a 5% decrease per period.
                for i in range(1, self.forecast_length):
                    forecast[i] = forecast[i - 1] * (1 - 0.05)
        # In trending conditions (ADX >= adx_threshold), no RSI-based trade is signaled,
        # so the forecast remains flat at the last price.

        return forecast
    def preprocess(self):
        pass
    
    def evaluate(self, data):
        return data