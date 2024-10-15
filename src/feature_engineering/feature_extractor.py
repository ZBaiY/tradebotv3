import ta
import pandas as pd
import numpy as np
import json
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.data_handler import DataHandler, RealTimeDataHandler
from src.feature_engineering.feature_selector import TopSelector, CorrelationSelector, HybridSelector, VolatilityBasedSelector

class FeatureExtractor:
    """
    A feature extractor that incrementally updates technical indicators for real-time trading.
    Reads indicator settings from a JSON file.
    """
    
    def __init__(self, data_handler=None):
        self.maximum_history = 100  # Maximum history to keep for indicators
        self.data_handler = data_handler
        self.symbols = self.data_handler.symbols
        self.settings = self._load_settings()
        self._initialize_indicators()
        self.indicators = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.select_method = None
        self.selections = []

    def _load_settings(self) -> dict:
        """
        Load indicator settings from a JSON file.
        :return: A dictionary containing settings for the indicators.
        """
        file_name = f'config/feature_set_{self.interval}.json'
        try:
            with open(file_name, 'r') as file:
                settings = json.load(file)
            return settings
        except FileNotFoundError:
            raise Exception(f"Settings file {file_name} not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON file {file_name}.")
        
        
    def _initialize_indicators(self):
        """
        Initialize state for all indicators based on settings from the JSON file.
        These initializations maintain internal state for efficient updates.
        """
        # Load settings from the JSON file
        # Load settings for momentum indicators
        self.rsi_period = self.settings.get("rsi_period", 14)  # Default RSI period is 14
        self.macd_short = self.settings.get("macd_short", 12)  # MACD short period
        self.macd_long = self.settings.get("macd_long", 26)    # MACD long period
        self.macd_signal = self.settings.get("macd_signal", 9) # MACD signal period
        self.stoch_period = self.settings.get("stoch_period", 14)  # Stochastic Oscillator period
        self.stoch_smooth_k = self.settings.get("stoch_smooth_k", 3) # %K smoothing for Stochastic Oscillator
        self.stoch_smooth_d = self.settings.get("stoch_smooth_d", 3) # %D smoothing for Stochastic Oscillator
        
         # Load settings for volatility indicators
        self.bollinger_period = self.settings.get("bollinger_period", 20)  # Default Bollinger Bands period
        self.atr_period = self.settings.get("atr_period", 14)  # Default ATR period

        # Load settings for volume indicators
        self.vwap_period = self.settings.get("vwap_period", 14)  # Default VWAP period
        self.obv_lookback = self.settings.get("obv_lookback", 100)

        # Load settings for trend indicators
        self.sma_period = self.settings.get("sma_period", 20)
        self.ema_period = self.settings.get("ema_period", 14)
        self.adx_period = self.settings.get("adx_period", 14)

        # Load settings for custom indicators
        self.custom_indicator_param = self.settings.get("custom_indicator_param", 5)

        file_name_window = f'config/fetch_real_time.json'
        try:
            with open(file_name_window, 'r') as file:
                settings = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Settings file {file_name_window} not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON file {file_name_window}.")
        
        memorysetting= settings.get("memory_setting",{"window_size": 100})
        self.maximum_history = memorysetting.get("window_size", 100)

        self.select_method = settings.get("select_method", "top_n")



    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initializes momentum indicators for the initial historical data.
        
        :param data: Historical data containing 'open', 'high', 'low', 'close', 'volume'.
        :return: DataFrame containing initialized momentum indicators.
        """
        # Create an empty DataFrame with the same index as the original data
        momentum_df = pd.DataFrame(index=data.index)

        # Add RSI to the new DataFrame
        momentum_df['rsi'] = ta.momentum.RSIIndicator(
            close=data['close'], window=self.rsi_period).rsi()

        # Add MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            close=data['close'],
            window_slow=self.macd_long,
            window_fast=self.macd_short,
            window_sign=self.macd_signal
        )
        momentum_df['macd'] = macd.macd()
        momentum_df['macd_signal'] = macd.macd_signal()
        momentum_df['macd_diff'] = macd.macd_diff()

        # Add Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(
            high=data['high'], 
            low=data['low'], 
            close=data['close'], 
            window=self.stoch_period, 
            smooth_window=self.stoch_smooth_k
        )
        momentum_df['stoch_k'] = stochastic.stoch()
        momentum_df['stoch_d'] = stochastic.stoch_signal()

        if len(momentum_df) > self.maximum_history:
            momentum_df = momentum_df.iloc[-self.maximum_history:]

        return momentum_df
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volatility indicators (e.g., Bollinger Bands, ATR) using the initialized parameters.
        """
        # Create a new DataFrame for volatility indicators
        volatility_df = pd.DataFrame(index=data.index)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['close'], window=self.bollinger_period)
        volatility_df['bollinger_mavg'] = bollinger.bollinger_mavg()
        volatility_df['bollinger_upper'] = bollinger.bollinger_hband()
        volatility_df['bollinger_lower'] = bollinger.bollinger_lband()
        
        
        # Average True Range (ATR)
        volatility_df['atr'] = ta.volatility.AverageTrueRange(
            high=data['high'], low=data['low'], close=data['close'], window=self.atr_period
        ).average_true_range()
        
        return volatility_df
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volume-based indicators (VWAP, OBV) using the initialized parameters.
        """
        # Create a new DataFrame for volume indicators
        volume_df = pd.DataFrame(index=data.index)
        
        # VWAP (Volume Weighted Average Price)
        volume_df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], 
            window=self.vwap_period  # If VWAP needs a custom period
        ).volume_weighted_average_price()

        # On-Balance Volume (OBV)
        volume_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=data['close'], volume=data['volume']
        ).on_balance_volume()

        return volume_df
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds trend-based indicators like Moving Averages (SMA, EMA) and ADX.
        """
        # Create a new DataFrame for trend indicators
        trend_df = pd.DataFrame(index=data.index)

        # Simple Moving Average (SMA)
        trend_df['sma'] = ta.trend.SMAIndicator(
            close=data['close'], window=self.sma_period
        ).sma_indicator()

        # Exponential Moving Average (EMA)
        trend_df['ema'] = ta.trend.EMAIndicator(
            close=data['close'], window=self.ema_period
        ).ema_indicator()

        # Average Directional Index (ADX)
        trend_df['adx'] = ta.trend.ADXIndicator(
            high=data['high'], low=data['low'], close=data['close'], window=self.adx_period
        ).adx()

        return trend_df
    
    def add_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds custom indicators based on specific calculations or parameters.
        """
        """
        # Create a new DataFrame for custom indicators
        custom_df = pd.DataFrame(index=data.index)
        
        # Example custom indicator logic (this is just a placeholder for actual custom logic)


        custom_df['custom_indicator'] = data['close'] * self.custom_indicator_param  # Placeholder formula

        return custom_df
        """
        pass
    
    def pre_run_indicators(self):
        """
        Initializes the indicators for the buffered data.
        """
        for symbol in self.symbols:
            data = self.data_handler.get_data(symbol, clean=True)
            momentum_df = self.add_momentum_indicators(data)
            volatility_df = self.add_volatility_indicators(data)
            volume_df = self.add_volume_indicators(data)
            trend_df = self.add_trend_indicators(data)
            custom_df = self.add_custom_indicators(data)
            self.indicators[symbol] = pd.concat([momentum_df, volatility_df, volume_df, trend_df, custom_df], axis=1)
            
        if self.select_method == "top_n":
            selector = TopSelector(self.indicators[self.symbols[0]], self.backtest_results)
            self.selections = selector.select_indicators()
        elif self.select_method == "correlation":
            selector = CorrelationSelector(self.indicators[self.symbols[0]])
            self.selections = selector.select_indicators()
        elif self.select_method == "hybrid":
            selector = HybridSelector(self.indicators[self.symbols[0]])
            self.selections = selector.select_indicators()
        elif self.select_method == "volatility":
            selector = VolatilityBasedSelector(self.indicators[self.symbols[0]])
            self.selections = selector.select_indicators()
        else:
            self.selections = self.indicators[self.symbols[0]].columns.tolist()
        for symbol in self.symbols:
            self.indicators[symbol] = self.indicators[symbol][self.selections]




############## Real-time update functions #################
    def update_indicators(self, new_data):
        new_datetime = new_data.pop('datetime')
        for symbol in self.symbols:
            new_row = pd.DataFrame({col: 0 for col in self.indicators[symbol].columns}, index=[new_datetime])
            self.indicators[symbol] = pd.concat([self.indicators[symbol], new_row])
            self.update_mom_indicators(symbol, new_datetime, new_data)
            self.update_vol_indicators(symbol, new_datetime)
            self.update_trend_indicators(symbol, new_datetime)
            self.update_custom_indicators(symbol, new_datetime)


    def update_mom_indicators(self, symbol, new_datetime, data):
        if 'rsi' in self.indicators[symbol].columns:
            self.indicators[symbol].at[new_datetime, 'rsi'] = self.update_rsi(symbol)
        if 'macd' in self.indicators[symbol].columns:
            prev_macd = {
                'short_ema': self.indicators[symbol]['macd'].iloc[-2],
                'long_ema': self.indicators[symbol]['macd'].iloc[-2],
                'macd_signal': self.indicators[symbol]['macd_signal'].iloc[-2]
            }
            new_close = data['close']
            updated_macd = self.update_macd(prev_macd, new_close)
            self.indicators[symbol].at[new_datetime, 'macd'] = updated_macd['macd_value']
            self.indicators[symbol].at[new_datetime, 'macd_signal'] = updated_macd['macd_signal']
            self.indicators[symbol].at[new_datetime, 'macd_diff'] = updated_macd['macd_diff']
        if 'stoch_k' in self.indicators[symbol].columns and 'stoch_d' in self.indicators[symbol].columns:
            prev_stochastic = {
                'highest_high': self.indicators[symbol]['stoch_k'].iloc[-2],
                'lowest_low': self.indicators[symbol]['stoch_k'].iloc[-2],
                'stoch_d': self.indicators[symbol]['stoch_d'].iloc[-2]
            }
            updated_stochastic = self.update_stochastic(prev_stochastic, data)
            self.indicators[symbol].at[new_datetime, 'stoch_k'] = updated_stochastic['stoch_k']
            self.indicators[symbol].at[new_datetime, 'stoch_d'] = updated_stochastic['stoch_d']

                
    def update_vol_indicators(self, symbol, new_datetime):
        pass

    def update_trend_indicators(self, symbol, new_datetime):
        pass

    def update_custom_indicators(self, symbol, new_datetime):
        pass

    def update_rsi(self, symbol):
        """
        Updates the RSI for the given symbol using the last calculated values and the new closing price.

        :param symbol: The trading symbol to update the RSI for.
        :return: Updated RSI value for the symbol.
        """
        data_window = self.data_handler.get_data_limit(symbol, self.rsi_period + 1, clean=True)

        if len(data_window) < self.rsi_period + 1:
            raise ValueError(f"Not enough data to update RSI for {symbol}. Requires at least {self.rsi_period + 1} data points.")

        close_diff = data_window['close'].diff().dropna()
        gains = close_diff.clip(lower=0)
        losses = -close_diff.clip(upper=0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()

        # Calculate the Relative Strength (RS)
        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss
        # Calculate the new RSI value
        new_rsi = 100 - (100 / (1 + rs))

        return new_rsi

    def update_macd(self, prev_macd, new_close):
        """
        Updates MACD using the last calculated values and the new closing price.
        
        :param prev_macd: Dictionary containing previous short EMA, long EMA, and signal line values.
        :param new_close: Latest closing price.
        :return: Updated MACD values.
        """
        ema_short = (new_close - prev_macd['short_ema']) * (2 / (self.macd_short + 1)) + prev_macd['short_ema']
        ema_long = (new_close - prev_macd['long_ema']) * (2 / (self.macd_long + 1)) + prev_macd['long_ema']
        macd_value = ema_short - ema_long
        macd_signal = (macd_value - prev_macd['macd_signal']) * (2 / (self.macd_signal + 1)) + prev_macd['macd_signal']
        macd_diff = macd_value - macd_signal

        return {
            'short_ema': ema_short,
            'long_ema': ema_long,
            'macd_value': macd_value,
            'macd_signal': macd_signal,
            'macd_diff': macd_diff
        }

    def update_stochastic(self, prev_stochastic, data):
        """
        Updates the Stochastic Oscillator using the new high, low, and close prices.
        
        :param prev_stochastic: Dictionary containing the highest high, lowest low, and last K/D values.
        :param new_high: Latest high price.
        :param new_low: Latest low price.
        :param new_close: Latest closing price.
        :return: Updated Stochastic Oscillator values.
        """
        new_high = data['high']
        new_low = data['low']
        new_close = data['close']
        highest_high = max(prev_stochastic['highest_high'], new_high)
        lowest_low = min(prev_stochastic['lowest_low'], new_low)

        stoch_k = ((new_close - lowest_low) / (highest_high - lowest_low)) * 100 if highest_high != lowest_low else 100
        stoch_d = (stoch_k + prev_stochastic['stoch_d'] * (self.stoch_smooth_k - 1)) / self.stoch_smooth_k

        return {
            'highest_high': highest_high,
            'lowest_low': lowest_low,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }
