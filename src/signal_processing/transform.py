import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""Transformers here won't bring NaN values, unlike filters which may filter out some entries."""

class TransformerBase:
    def __init__(self):
        pass
    def transform(self, data):
        """Transform method to be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class ReturnTransformer(TransformerBase):
    def __init__(self):
        super().__init__()
        self.lookback = 2

    def transform(self, data):
        """Transform price data into simple returns."""
        data_copy = data.copy()
        data_copy = data_copy.pct_change().fillna(0)
        return data_copy
    
    def on_new_data(self, two_data):
        """Transform the new data."""
        if len(two_data) < 2:
            raise ValueError("Input data must contain at least two data points.")
        n_return = (two_data / two_data.shift(1) - 1).fillna(0)
        return n_return.iloc[1]

class LogReturnTransformer(TransformerBase):
    def __init__(self):
        super().__init__()
        self.lookback = 2 # need two data points for log return update

    def transform(self, data):
        """Transform price data into log returns."""
        result = {}
        data_copy = data.copy()
        data_copy = np.log(data_copy / data_copy.shift(1)).fillna(0)
        result = data_copy
        return result
    def on_new_data(self, two_data):
        """Transform the new data."""
        if len(two_data) < 2:
            raise ValueError("Input data must contain at least two data points.")
        log_return = np.log(two_data / two_data.shift(1)).fillna(0)
        return log_return.iloc[1]

class FourierTransformer(TransformerBase):
    def __init__(self):
        super().__init__()
        self.lookback = 'all'
        """
        The result of the Fourier Transform is a series of complex numbers.
        For a real-valued input signal of length N, the frequencies represented by each index
        are from 0 up to the Nyquist frequency (the highest frequency that can be represented)
        which occurs at index N/2
        Beyond the Nyquist frequency, the frequency mirror the lower half.

        the first half of the FFT result contains the positive frequencies low to high,
        0Hz(The DC) to N/2 Hz,
        while the second half contains the negative frequencies, high to low,
        -N/2+1 Hz to -1 Hz.
        (no 0Hz/the DC) the result[1:N] is symmetric with result[N+1:2*N-1].
        """
    def transform(self, data):
        """Apply Fourier Transform (FFT) to the data."""
        data_copy = data.copy()
        if isinstance(data_copy, pd.DataFrame) or isinstance(data_copy, pd.Series):
            # Drop NaN values as FFT does not handle them
            data_cleaned = data_copy.dropna()
            data_flattened = data_cleaned.values.flatten()
            if len(data_cleaned) == 0:
                raise ValueError("No valid data points after dropping NaN values.")
            fft_result = np.fft.fft(data_flattened)
            return pd.Series(fft_result, index=range(len(fft_result)))
        else:
            # Handle raw numpy arrays or other data formats
            if np.isnan(data_copy).any():
                raise ValueError("Data contains NaN values.")
            fft_result = np.fft.fft(data_copy)
            return pd.Series(fft_result, index=range(len(fft_result)))
    
    def inverse_transform(self, data):
        """Apply Inverse Fourier Transform (IFFT) to the frequency space data."""
        data_copy = data.copy()
        if isinstance(data_copy, pd.DataFrame) or isinstance(data_copy, pd.Series):
            data_cleaned = data_copy.dropna()
            ifft_result = np.fft.ifft(data_cleaned)
            return pd.Series(ifft_result, index=range(len(ifft_result)))
        else:
            ifft_result = np.fft.ifft(data_copy)
            return pd.Series(ifft_result, index=range(len(ifft_result)))

    

### ScalerTranformer Cannot be followed by ReturnTransformer or LogReturnTransformer, because of zeros ########
class ScalerTransformer(TransformerBase):
    def __init__(self, symbols=None, scaler=None):
        super().__init__()
        self.symbols = symbols
        self.lookback = 1 # only one data point at a time for updating
        if scaler is None:
            raise ValueError("A scaler object must be provided.")
        elif scaler == 'minmax':
            self.scaler = {symbol: MinMaxScaler() for symbol in symbols}
        elif scaler == 'standard':
            self.scaler = {symbol: StandardScaler() for symbol in symbols}
    
    def fit_scaler(self, symbol, data):
        """Update the scaler with new data."""
        """Will be used as initilization, as well as for updating the scaler per week."""
        self.scaler[symbol].fit(data.values.reshape(-1, 1))

    def on_new_data(self, symbol, new_data):
        """transform the new data."""
        data_copy = new_data.copy()
        transformed_data = self.scaler[symbol].transform(data_copy.values.reshape(-1, 1))
        return pd.DataFrame(transformed_data, index=new_data.index)

    def transform(self, symbol, data):
        """Apply the scaler to the data."""
        data_copy = data.copy()
        transformed_data = self.scaler[symbol].transform(data_copy.values.reshape(-1, 1))
        return pd.DataFrame(transformed_data, index=data.index)
        







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

