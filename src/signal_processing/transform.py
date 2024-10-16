import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TransformerBase:
    def __init__(self, symbols=None):
        self.symbols = symbols

    def transform(self, data):
        """Transform method to be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class ReturnTransformer(TransformerBase):
    def __init__(self, symbols=None):
        super().__init__(symbols)

    def transform(self, data):
        """Transform price data into simple returns."""
        result = {}
        for symbol in self.symbols:
            data_copy = data[symbol].copy()
            data_copy = data_copy.pct_change().fillna(0)
            result[symbol] = data_copy
        return result

class LogReturnTransformer(TransformerBase):
    def __init__(self, symbols=None):
        super().__init__(symbols)

    def transform(self, data):
        """Transform price data into log returns."""
        result = {}
        for symbol in self.symbols:
            data_copy = data[symbol].copy()
            data_copy = np.log(data_copy / data_copy.shift(1)).fillna(0)
            result[symbol] = data_copy
        return result
    def on_new_data(self, new_data):
        """Transform the new data."""
        return np.log(new_data / new_data.shift(1)).fillna(0)

class FourierTransformer(TransformerBase):
    def __init__(self, symbols=None):
        super().__init__(symbols)
        
    def transform(self, data):
        """Apply Fourier Transform (FFT) to the data."""
        result = {}
        for symbol in self.symbols:
            data_copy = data[symbol].copy()
            if isinstance(data_copy, pd.DataFrame) or isinstance(data_copy, pd.Series):
                # Drop NaN values as FFT does not handle them
                data_cleaned = data_copy.dropna()
                fft_result = np.fft.fft(data_cleaned)
                result[symbol] = pd.Series(fft_result, index=range(len(fft_result)))
            else:
                if np.isnan(data_copy).any():
                    raise ValueError(f"Data for symbol {symbol} contains NaN values.")
                fft_result = np.fft.fft(data_copy)
                result[symbol] = pd.Series(fft_result, index=range(len(fft_result)))
                result[symbol] = fft_result
        return result
    
    def inverse_transform(self, data):
        """Apply Inverse Fourier Transform (IFFT) to the frequency space data."""
        result = {}
        for symbol in self.symbols:
            data_copy = data[symbol].copy()
            if isinstance(data_copy, pd.DataFrame) or isinstance(data_copy, pd.Series):
                data_cleaned = data_copy.dropna()
                ifft_result = np.fft.ifft(data_cleaned)
                result[symbol] = pd.Series(ifft_result, index=data_cleaned.index)
            else:
                ifft_result = np.fft.ifft(data_copy)
                result[symbol] = ifft_result
        return result
    


class ScalerTransformer(TransformerBase):
    def __init__(self, symbols=None, scaler=None):
        super().__init__(symbols)
        if scaler is None:
            raise ValueError("A scaler object must be provided.")
        elif scaler == 'minmax':
            self.scaler = {symbol: MinMaxScaler() for symbol in symbols}
        elif scaler == 'standard':
            self.scaler = {symbol: StandardScaler() for symbol in symbols}
        
    def fit_scaler(self, data):
        """Update the scaler with new data."""
        """Will be used as initilization, as well as for updating the scaler per week."""
        for symbol in self.symbols:
            self.scaler[symbol].fit(data[symbol].values.reshape(-1, 1))

    def on_new_data(self, new_data):
        """transform the new data."""
        result = {}
        for symbol in self.symbols:
            data_copy = new_data[symbol].copy()
            result[symbol] = self.scaler[symbol].transform(data_copy.values.reshape(-1, 1))
        return result
    def transform(self, data):
        """Apply the scaler to the data."""
        result = {}
        for symbol in self.symbols:
            data_copy = data[symbol].copy()
            result[symbol] = self.scaler[symbol].transform(data_copy.values.reshape(-1, 1))
        return result
    







class CustomTransformer(TransformerBase):
    def __init__(self, custom_function):
        """
        Custom transformer allows any arbitrary transformation function.
        :param custom_function: A lambda or function to apply to the data
        """
        pass
    def transform(self, data):
        """Apply the custom function to the data."""
        pass

