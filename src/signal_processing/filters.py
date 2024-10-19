
"""Filters may bring NaN values, like moving average filters will bring something in the beginning."""


class FilterBase:
    def __init__(self):
        self.lookback = 1 
        self.rolling_window = 1
    def apply(self, data):
        raise NotImplementedError("This method should be overridden by subclasses.")


class MovingAverageFilter(FilterBase):
    def __init__(self, lookback_size=5):
        super().__init__()
        self.rolling_window = lookback_size
        self.lookback = lookback_size
        

    def apply(self, data):
        data_copy = data.copy()
        data_copy = data_copy.rolling(window=self.rolling_window).mean().dropna()
        return data_copy
    
    """def on_new_data(self, new_data):
        data_copy = new_data.copy()
        data_copy = data_copy.rolling(window=self.lookback).mean()
        return data_copy"""

class ExponentialSmoothingFilter(FilterBase):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.rolling_window = 1
        self.alpha = alpha
        self.lookback = 2

    def apply(self, data):
        data_copy = data.copy()
        data_copy = data_copy.ewm(alpha=self.alpha).mean()
        return data_copy
    
    """def on_new_data(self, new_data):
        data_copy = new_data.copy()
        data_copy = data_copy.ewm(alpha=self.alpha).mean()
        return data_copy"""
