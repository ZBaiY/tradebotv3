import ta
import pandas as pd
import json
class FeatureExtractor:
    """
    A feature extractor that incrementally updates technical indicators for real-time trading.
    Reads indicator settings from a JSON file.
    """
    
    def __init__(self):
        self.maximum_history = 100  # Maximum history to keep for indicators
        self.settings = self._load_settings()
        self._initialize_indicators()

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

import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler

class FeatureExtractorRealTime(FeatureExtractor):
    def __init__(self, data_handler: RealTimeDataHandler):
        super().__init__()
        self.data_handler = data_handler

        # Use get_data() function in realtimedatahandler to get the latest data