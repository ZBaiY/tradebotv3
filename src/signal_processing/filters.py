class FilterBase:
    def __init__(self, symbols=None):
        self.symbols = symbols

    def apply(self, data):
        raise NotImplementedError("This method should be overridden by subclasses.")


class MovingAverageFilter(FilterBase):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size

    def apply(self, data):
        data_copy = data.copy()
        for symbol in self.symbols:
            data_copy[symbol] = data_copy[symbol].rolling(window=self.window_size).mean()
        return data_copy

class ExponentialSmoothingFilter(FilterBase):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def apply(self, data):
        data_copy = data.copy()
        for symbol in self.symbols:
            data_copy[symbol] = data_copy[symbol].ewm(alpha=self.alpha).mean()
        return data_copy
    
