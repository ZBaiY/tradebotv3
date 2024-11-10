import ta
import pandas as pd
import numpy as np
import json
import sys
import gc
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
from src.feature_engineering.feature_selector import TopSelector, CorrelationSelector, HybridSelector, VolatilityBasedSelector
from datetime import datetime, timezone
import time


class FeatureExtractor:
    """
    A feature extractor that incrementally updates technical indicators for real-time trading.
    Reads indicator settings from a JSON file.
    """
    
    def __init__(self, data_handler=None):
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        self.maximum_history = 100  # Maximum history to keep for indicators
        self.data_handler = data_handler
        self.symbols = self.data_handler.symbols
        self.interval = self.data_handler.interval_str
        self.settings = self._load_settings()
        self._initialize_indicators()
        self.indicators = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.helpers = {symbol: pd.DataFrame() for symbol in self.symbols}

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
        self.rsi_period = self.settings.get("rsi_period", None)  # Default RSI period is 14
        self.macd_short = self.settings.get("macd_short", None)  # MACD short period
        self.macd_long = self.settings.get("macd_long", 26)    # MACD long period
        self.macd_signal = self.settings.get("macd_signal", 9) # MACD signal period
        self.stoch_period = self.settings.get("stoch_period", None)  # Stochastic Oscillator period
        self.stoch_smooth_k = self.settings.get("stoch_smooth_k", 3) # %K smoothing for Stochastic Oscillator
        self.stoch_smooth_d = self.settings.get("stoch_smooth_d", 3) # %D smoothing for Stochastic Oscillator
        
        # Load settings for volatility indicators
        self.bollinger_period = self.settings.get("bollinger_period", None)  # Default Bollinger Bands period
        self.bollinger_std = self.settings.get("bollinger_std", 2)  # Default Bollinger Bands standard deviation
        self.atr_period = self.settings.get("atr_period", None)  # Default ATR period

        # Load settings for volume indicators
        self.vwap_period = self.settings.get("vwap_period", None)  # Default VWAP period
        self.obv_lookback = self.settings.get("obv_lookback", None)

        # Load settings for trend indicators
        self.sma_period = self.settings.get("sma_period", None)
        self.ema_period = self.settings.get("ema_period", None)
        self.adx_period = self.settings.get("adx_period", None)

        # Load settings for custom indicators
        self.custom_indicator_param = self.settings.get("custom_indicator_param", 5)

        file_name_window = f'config/fetch_real_time.json'
        try:
            with open(file_name_window, 'r') as file:
                self.settings = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Settings file {file_name_window} not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON file {file_name_window}.")
        
        self.maximum_history = self.data_handler.window_size
        self.select_method = self.settings.get("select_method", None)
        self.data_handler.subscribe(self)



    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        Initializes momentum indicators for the initial historical data.
        
        :param data: Historical data containing 'open', 'high', 'low', 'close', 'volume'.
        :return: DataFrame containing initialized momentum indicators.
        """
        # Create an empty DataFrame with the same index as the original data
        momentum_df = pd.DataFrame(index=data.index)
        momentum_hdf = pd.DataFrame(index=data.index)
        if self.rsi_period is not None:
        # Add RSI to the new DataFrame
            momentum_df['rsi'] = ta.momentum.RSIIndicator(
                close=data['close'], window=self.rsi_period).rsi().bfill().ffill()
            rsi_h = pd.DataFrame(index=data.index)
            rsi_h['change'] = data['close'].diff()
            rsi_h['gain'] = rsi_h['change'].apply(lambda x: x if x > 0 else 0)
            rsi_h['loss'] = rsi_h['change'].apply(lambda x: -x if x < 0 else 0)
            rsi_h['gain'] = rsi_h['change'].apply(lambda x: x if x > 0 else 0)
            rsi_h['loss'] = rsi_h['change'].apply(lambda x: -x if x < 0 else 0)
            momentum_hdf['avg_gain'] = rsi_h['gain'].rolling(window=self.rsi_period).mean()
            momentum_hdf['avg_loss'] = rsi_h['loss'].rolling(window=self.rsi_period).mean()


        # Add MACD (Moving Average Convergence Divergence)
        if self.macd_short is not None:
            macd = ta.trend.MACD(
                close=data['close'],
                window_slow=self.macd_long,
                window_fast=self.macd_short,
                window_sign=self.macd_signal
            )
            ###### It looks like MACD line need four columns, not economic ########
            momentum_df['macd'] = macd.macd().bfill().ffill()
            momentum_df['macd_signal'] = macd.macd_signal().bfill().ffill()
            momentum_df['macd_diff'] = macd.macd_diff().bfill().ffill()
            momentum_hdf['macd_long_ema'] = ta.trend.EMAIndicator(close=data['close'], window=self.macd_long).ema_indicator().bfill().ffill()

        # Add Stochastic Oscillator
        if self.stoch_period is not None:
            stochastic = ta.momentum.StochasticOscillator(
                high=data['high'], 
                low=data['low'], 
                close=data['close'], 
                window=self.stoch_period, 
                smooth_window=self.stoch_smooth_k
            )
            momentum_df['stoch_k'] = stochastic.stoch().bfill().ffill()
            momentum_df['stoch_d'] = stochastic.stoch_signal().bfill().ffill()
            momentum_hdf['lookback_high'] = data['high'].rolling(window=self.stoch_period).max()
            momentum_hdf['lookback_low'] = data['low'].rolling(window=self.stoch_period).min()

        if len(momentum_df) > self.maximum_history:
            momentum_df = momentum_df.tail(self.maximum_history)
            
        if len(momentum_hdf) > self.maximum_history:
            momentum_hdf = momentum_hdf.tail(self.maximum_history)
        return momentum_df, momentum_hdf
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volatility indicators (e.g., Bollinger Bands, ATR) using the initialized parameters.
        """
        # Create a new DataFrame for volatility indicators
        volatility_df = pd.DataFrame(index=data.index)
        if self.bollinger_period is not None:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=data['close'], window=self.bollinger_period, window_dev=self.bollinger_std)
            volatility_df['bollinger_mavg'] = bollinger.bollinger_mavg().bfill().ffill() # Middle band (SMA)
            volatility_df['bollinger_upper'] = bollinger.bollinger_hband().bfill().ffill()
            volatility_df['bollinger_lower'] = bollinger.bollinger_lband().bfill().ffill()
            
        if self.atr_period is not None:
            # Average True Range (ATR)
            volatility_df['atr'] = ta.volatility.AverageTrueRange(
                high=data['high'], low=data['low'], close=data['close'], window=self.atr_period
            ).average_true_range().bfill().ffill()
        if len(volatility_df) > self.maximum_history:
            volatility_df = volatility_df.tail(self.maximum_history)

        return volatility_df
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volume-based indicators (VWAP, OBV) using the initialized parameters.
        """
        # Create a new DataFrame for volume indicators
        volume_df = pd.DataFrame(index=data.index)
        if self.vwap_period is not None:
            # VWAP (Volume Weighted Average Price)
            volume_df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], 
                window=self.vwap_period  # If VWAP needs a custom period
            ).volume_weighted_average_price().bfill().ffill()
        if self.obv_lookback is not None:
            # On-Balance Volume (OBV)
            volume_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=data['close'], volume=data['volume']
            ).on_balance_volume().bfill().ffill()

        return volume_df
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds trend-based indicators like Moving Averages (SMA, EMA) and ADX.
        """
        # Create a new DataFrame for trend indicators
        
        trend_df = pd.DataFrame(index=data.index)
        if self.sma_period is not None:
            # Simple Moving Average (SMA)
            trend_df['sma'] = ta.trend.SMAIndicator(
                close=data['close'], window=self.sma_period
            ).sma_indicator().bfill().ffill()
        if self.ema_period is not None:
            # Exponential Moving Average (EMA)
            trend_df['ema'] = ta.trend.EMAIndicator(
                close=data['close'], window=self.ema_period
            ).ema_indicator().bfill().ffill()
        if self.adx_period is not None:
            # Average Directional Index (ADX)
            trend_df['adx'] = ta.trend.ADXIndicator(
                high=data['high'], low=data['low'], close=data['close'], window=self.adx_period
            ).adx().bfill().ffill()
        if len(trend_df) > self.maximum_history:
            trend_df = trend_df.tail(self.maximum_history)

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
            momentum_df, momentum_hdf = self.add_momentum_indicators(data)
            volatility_df = self.add_volatility_indicators(data)
            volume_df = self.add_volume_indicators(data)
            trend_df = self.add_trend_indicators(data)
            custom_df = self.add_custom_indicators(data)
            self.indicators[symbol] = pd.concat([momentum_df, volatility_df, volume_df, trend_df, custom_df], axis=1)
            self.helpers[symbol] = momentum_hdf
            del momentum_df, volatility_df, volume_df, trend_df, custom_df
            gc.collect()

        if self.select_method == "top_n":
            selector = TopSelector(self.indicators[self.symbols[0]], self.backtest_results)
            self.selections = selector.select_indicators()
        elif self.select_method == "correlation":
            corr_thr = self.settings.get("correlation_threshold", 0.7)
            selector = CorrelationSelector(self.indicators[self.symbols[0]], corr_thr)
            self.selections = selector.select_indicators()
        elif self.select_method == "hybrid":
            selector = HybridSelector(self.indicators[self.symbols[0]])
            self.selections = selector.select_indicators()
        elif self.select_method == "volatility":
            selector = VolatilityBasedSelector(self.indicators[self.symbols[0]])
            self.selections = selector.select_indicators()
        else: # no selection method specified
            self.selections = self.indicators[self.symbols[0]].columns.tolist()
        for symbol in self.symbols:
            self.indicators[symbol] = self.indicators[symbol][self.selections]
    




############## Real-time update functions #################
    def update(self, new_data):
        new_datetime = new_data[self.symbols[0]].index
        for symbol in self.symbols:
            data_symbol = new_data[symbol]
            new_row = pd.DataFrame({col: 0 for col in self.indicators[symbol].columns}, index=new_datetime)
            self.indicators[symbol] = pd.concat([self.indicators[symbol], new_row])
            self.helpers[symbol] = pd.concat([self.helpers[symbol], new_row])
            self.update_mom_indicators(symbol, data_symbol)
            self.update_vol_indicators(symbol, data_symbol)
            self.update_trend_indicators(symbol, data_symbol)
            self.update_volume_indicators(symbol, data_symbol)
            # self.update_custom_indicators(symbol, new_datetime, data_symbol)
            if len(self.indicators[symbol]) > self.maximum_history:
                self.indicators[symbol] = self.indicators[symbol].tail(self.maximum_history)


    def update_mom_indicators(self, symbol, data):
        last_index = self.indicators[symbol].index[-1]
        if 'rsi' in self.indicators[symbol].columns:
            prev_data = self.data_handler.get_data_limit(symbol, 2, clean=True)
            prev_close = prev_data['close'].iloc[-2]
            new_close = data['close'].iloc[0]
            prev_rsi = {
                'avg_gain': self.helpers[symbol]['avg_gain'].iloc[-2],
                'avg_loss': self.helpers[symbol]['avg_loss'].iloc[-2],
                'previous_close': prev_close
            }
            updated_rsi = self.update_rsi(prev_rsi, new_close)
            self.indicators[symbol].loc[last_index, 'rsi'] = updated_rsi['rsi_value']
            self.helpers[symbol].loc[last_index, 'avg_gain'] = updated_rsi['avg_gain']
            self.helpers[symbol].loc[last_index, 'avg_loss'] = updated_rsi['avg_loss']
            
        if 'macd' in self.indicators[symbol].columns:
            prev_macd = {
                'short_ema': self.helpers[symbol]['macd_long_ema'].iloc[-2]-self.indicators[symbol]['macd'].iloc[-2],
                'long_ema': self.helpers[symbol]['macd_long_ema'].iloc[-2],
                'macd_signal': self.indicators[symbol]['macd_signal'].iloc[-2]
            }
            new_close = data['close'].iloc[0]
            updated_macd = self.update_macd(prev_macd, new_close)
            
            self.indicators[symbol].loc[last_index, 'macd'] = updated_macd['macd_value']
            self.indicators[symbol].loc[last_index, 'macd_signal'] = updated_macd['macd_signal']
            self.indicators[symbol].loc[last_index, 'macd_diff'] = updated_macd['macd_diff']
            self.helpers[symbol].loc[last_index, 'macd_long_ema'] = updated_macd['long_ema']

        if 'stoch_k' in self.indicators[symbol].columns and 'stoch_d' in self.indicators[symbol].columns:
            prev_stochastic = {
                'highest_high': self.helpers[symbol]['lookback_high'].iloc[-2],
                'lowest_low': self.helpers[symbol]['lookback_low'].iloc[-2],
                'stoch_d': self.indicators[symbol]['stoch_d'].iloc[-2],
            }
            updated_stochastic = self.update_stochastic(
                prev_stochastic, data
            )
            # Update the momentum_df with the new stoch_k and stoch_d values
            self.indicators[symbol].loc[last_index, 'stoch_k'] = updated_stochastic['stoch_k']
            self.indicators[symbol].loc[last_index, 'stoch_d'] = updated_stochastic['stoch_d']
            self.helpers[symbol].loc[last_index, 'lookback_high'] = updated_stochastic['highest_high']
            self.helpers[symbol].loc[last_index, 'lookback_low'] = updated_stochastic['lowest_low']

                
    def update_vol_indicators(self, symbol, data):
        """
        Updates volatility indicators (Bollinger Bands, ATR) using the last calculated values and the new price data.
        
        :param symbol: The trading symbol to update.
        :param new_datetime: The timestamp of the new data.
        :param data: The latest data, containing 'close', 'high', 'low'.
        """
        # Update Bollinger Bands
        last_index = self.indicators[symbol].index[-1]
        if 'bollinger_mavg' in self.indicators[symbol].columns:
            prev_bollinger = {
                'sma': self.indicators[symbol]['bollinger_mavg'].iloc[-2], # Middle band (SMA)
                'stddev': (self.indicators[symbol]['bollinger_upper'].iloc[-2] - self.indicators[symbol]['bollinger_mavg'].iloc[-2]) / self.bollinger_std
            }
            new_close = data['close'].iloc[0]
            updated_bollinger = self.update_bollinger_bands(prev_bollinger, new_close)
            self.indicators[symbol].loc[last_index, 'bollinger_mavg'] = updated_bollinger['sma']
            self.indicators[symbol].loc[last_index, 'bollinger_upper'] = updated_bollinger['upper_band']
            self.indicators[symbol].loc[last_index, 'bollinger_lower'] = updated_bollinger['lower_band']
        
        # Update ATR
        if 'atr' in self.indicators[symbol].columns:
            prev_close = self.data_handler.get_data_limit(symbol, 2, clean=True)['close'].iloc[-2]
            prev_atr = {
                'atr': self.indicators[symbol]['atr'].iloc[-2],
                'prev_close': prev_close
            }
            updated_atr = self.update_atr(prev_atr, data['high'].iloc[0], data['low'].iloc[0], data['close'].iloc[0])
            self.indicators[symbol].loc[last_index, 'atr'] = updated_atr['atr']
    
    def update_volume_indicators(self, symbol, data):
        """
        Updates volume-based indicators (VWAP, OBV) using the last calculated values and the new price data.
        
        :param symbol: The trading symbol to update.
        :param new_datetime: The timestamp of the new data.
        :param data: The latest data, containing 'close', 'volume'.
        """
        # Update Volume Weighted Average Price (VWAP)
        last_index = self.indicators[symbol].index[-1]
        if 'vwap' in self.indicators[symbol].columns:
            prev_vwap = self.indicators[symbol]['vwap'].iloc[-2]
            new_close = data['close'].iloc[0]
            new_volume = data['volume'].iloc[0]
            updated_vwap = self.update_vwap(prev_vwap, new_close, new_volume)
            self.indicators[symbol].loc[last_index, 'vwap'] = updated_vwap

        # Update On-Balance Volume (OBV)
        if 'obv' in self.indicators[symbol].columns:
            prev_obv = self.indicators[symbol]['obv'].iloc[-2]
            prev_close = self.data_handler.get_data_limit(symbol, 2, clean=True)['close'].iloc[-2]
            new_close = data['close'].iloc[0]
            new_volume = data['volume'].iloc[0]
            updated_obv = self.update_obv(prev_obv,prev_obv, new_close, new_volume)
            self.indicators[symbol].loc[last_index, 'obv'] = updated_obv

    def update_trend_indicators(self, symbol, data):
        """
        Updates trend-based indicators (SMA, EMA, ADX) using the last calculated values and the new price data.

        :param symbol: The trading symbol to update.
        :param new_datetime: The timestamp of the new data.
        :param data: The latest data, containing 'close', 'high', 'low'.
        """
        # Update Simple Moving Average (SMA)
        last_index = self.indicators[symbol].index[-1]
        if 'sma' in self.indicators[symbol].columns:
            prev_sma = self.indicators[symbol]['sma'].iloc[-2]
            new_close = data['close'].iloc[0]
            updated_sma = self.update_sma(prev_sma, new_close)
            self.indicators[symbol].loc[last_index, 'sma'] = updated_sma

        # Update Exponential Moving Average (EMA)
        if 'ema' in self.indicators[symbol].columns:
            prev_ema = self.indicators[symbol]['ema'].iloc[-2]
            new_close = data['close'].iloc[0]
            updated_ema = self.update_ema(prev_ema, new_close)
            self.indicators[symbol].loc[last_index, 'ema'] = updated_ema

        # Update Average Directional Index (ADX)
        if 'adx' in self.indicators[symbol].columns:
            prev_adx = self.indicators[symbol]['adx'].iloc[-2]
            prev_data = self.data_handler.get_data_limit(symbol, 2, clean=True).iloc[-2]
            prev_high = prev_data['high']
            prev_low = prev_data['low']
            prev_close = prev_data['close']
            current_high = data['high'].iloc[0]
            current_low = data['low'].iloc[0]
            updated_adx = self.update_adx(prev_high, prev_low, prev_close, current_high, current_low, prev_adx)
            self.indicators[symbol].loc[last_index, 'adx'] = updated_adx
    
    def update_custom_indicators(self, symbol):
        pass
############## Momentum Indicator Functions #################
    def update_rsi(self, prev_rsi, new_close):
        # Calculate the new gain or loss
        new_diff = new_close - prev_rsi['previous_close']
        gain = max(new_diff, 0)
        loss = abs(min(new_diff, 0))

        # Incrementally update the average gain and average loss
        avg_gain = (prev_rsi['avg_gain'] * (self.rsi_period - 1) + gain) / self.rsi_period
        avg_loss = (prev_rsi['avg_loss'] * (self.rsi_period - 1) + loss) / self.rsi_period

        # Calculate the updated RSI value
        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return {
            'rsi_value': rsi,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss
        }

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
        new_high = data['high'].iloc[0]
        new_low = data['low'].iloc[0]
        new_close = data['close'].iloc[0]
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
############## Volatility Indicator Functions #################
    def update_bollinger_bands(self, prev_bollinger, new_close):
        """
        Updates Bollinger Bands using the last moving average and standard deviation.
        
        :param prev_bollinger: Dictionary containing the last simple moving average and standard deviation.
        :param new_close: Latest closing price.
        :return: Updated Bollinger Bands values.
        """
        # Update the moving average (SMA)
        sma = (prev_bollinger['sma'] * (self.bollinger_period - 1) + new_close) / self.bollinger_period

        # Update the rolling standard deviation (stddev)
        variance = ((prev_bollinger['stddev'] ** 2) * (self.bollinger_period - 1) +
                    (new_close - sma) ** 2) / self.bollinger_period
        stddev = np.sqrt(variance)

        # Calculate Bollinger Bands
        upper_band = sma + (2 * stddev)
        lower_band = sma - (2 * stddev)

        return {
            'sma': sma,
            'stddev': stddev,
            'upper_band': upper_band,
            'lower_band': lower_band
        }

    def update_atr(self, prev_atr, new_high, new_low, new_close):
        """
        Updates the Average True Range (ATR) using the last ATR value and the new high, low, and close prices.
        
        :param prev_atr: Dictionary containing the last ATR and previous closing price.
        :param new_high: Latest high price.
        :param new_low: Latest low price.
        :param new_close: Latest closing price.
        :return: Updated ATR value.
        """
        # Calculate the new True Range (TR)
        tr = max(new_high - new_low, abs(new_high - prev_atr['prev_close']), abs(new_low - prev_atr['prev_close']))

        # Update the ATR using the exponential moving average formula
        new_atr = (prev_atr['atr'] * (self.atr_period - 1) + tr) / self.atr_period

        return {
            'atr': new_atr,
            'prev_close': new_close
        }

    ############## Trend Indicator Functions #################
    def update_sma(self, prev_sma, new_close):
        """
        Updates the Simple Moving Average (SMA) using the previous SMA and new closing price.

        :param prev_sma: The last calculated SMA value.
        :param new_close: The latest closing price.
        :return: Updated SMA value.
        """
        updated_sma = (prev_sma * (self.sma_period - 1) + new_close) / self.sma_period
        return updated_sma

    def update_ema(self, prev_ema, new_close):
        """
        Updates the Exponential Moving Average (EMA) using the previous EMA and new closing price.

        :param prev_ema: The last calculated EMA value.
        :param new_close: The latest closing price.
        :return: Updated EMA value.
        """
        alpha = 2 / (self.ema_period + 1)
        updated_ema = (new_close - prev_ema) * alpha + prev_ema
        return updated_ema

    def update_adx(self, prev_high, prev_low, prev_close, current_high, current_low, prev_adx):
        """
        Updates the Average Directional Index (ADX) using the previous ADX and new high, low, close prices.

        :param symbol: The trading symbol.
        :param prev_adx: The last calculated ADX value.
        :param data: Latest data, containing 'high', 'low', 'close'.
        :return: Updated ADX value.
        """
        # Retrieve the previous data points for the indicator calculation
        tr = max(current_high - current_low, abs(current_high - prev_close), abs(current_low - prev_close))
        # Step 2: Calculate +DM and -DM
        up_move = current_high - prev_high
        down_move = prev_low - current_low
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0
        # Step 3: Calculate +DI and -DI (Directional Indicators)
        plus_di = (plus_dm / tr) * 100 if tr != 0 else 0
        minus_di = (minus_dm / tr) * 100 if tr != 0 else 0
        # Using ta-lib to calculate ADX for the current data
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
        adx = ((prev_adx * (self.adx_period - 1)) + dx) / self.adx_period

        return adx
    

    def update_vwap(self, prev_vwap, new_close, new_volume):
        """
        Updates the Volume Weighted Average Price (VWAP) using the previous VWAP, new closing price, and volume.

        :param prev_vwap: The last calculated VWAP value.
        :param new_close: The latest closing price.
        :param new_volume: The latest trading volume.
        :return: Updated VWAP value.
        """
        updated_vwap = ((prev_vwap * (self.vwap_period - 1)) + (new_close * new_volume)) / self.vwap_period
        return updated_vwap
    
    def update_obv(self, prev_obv, prev_close, new_close, new_volume):
        """
        Updates the On-Balance Volume (OBV) using the previous OBV, new closing price, and volume.

        :param prev_obv: The last calculated OBV value.
        :param new_close: The latest closing price.
        :param new_volume: The latest trading volume.
        :return: Updated OBV value.
        """
        if new_close > prev_close:
            updated_obv = prev_obv + new_volume
        elif new_close < prev_close:
            updated_obv = prev_obv - new_volume
        else:
            updated_obv = prev_obv
        return updated_obv
    
    def get_last_indicator(self, symbol, indicator):
        """
        Get the last value of a specific indicator for a given symbol.
        
        :param symbol: The trading symbol.
        :param indicator: The indicator to retrieve.
        :return: The last value of the indicator.
        """
        return self.indicators[symbol][indicator].iloc[-1]


if __name__ == "__main__":
    data_handler = RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')
    next_fetch_time,last_fetch_time = data_handler.pre_run_data()
    ft_ext = FeatureExtractor(data_handler)
    ft_ext.pre_run_indicators()

    is_running = True
    while is_running:
        new_data = data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
        # print(new_data)
        # input("Press Enter to continue...")
        now = datetime.now(timezone.utc)
        data_handler.notify_subscribers(new_data)
        next_fetch_time = data_handler.calculate_next_grid(now)
        # print(ft_ext.indicators['BTCUSDT'].tail())
        sleep_duration = (next_fetch_time - now).total_seconds()+1 # Add 1 second to avoid fetching data too early, total seconds rounds down
        # print(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
        time.sleep(sleep_duration)