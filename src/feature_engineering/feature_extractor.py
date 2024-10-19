import ta
import pandas as pd
import numpy as np
import json
import sys
import gc
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
                settings = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Settings file {file_name_window} not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON file {file_name_window}.")
        
        memorysetting= settings.get("memory_setting",{"window_size": 100})
        self.maximum_history = memorysetting.get("window_size", 100)

        self.select_method = settings.get("select_method", "top_n")
        self.data_handler.subscribe(self)



    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initializes momentum indicators for the initial historical data.
        
        :param data: Historical data containing 'open', 'high', 'low', 'close', 'volume'.
        :return: DataFrame containing initialized momentum indicators.
        """
        # Create an empty DataFrame with the same index as the original data
        momentum_df = pd.DataFrame(index=data.index)
        if self.rsi_period is not None:
        # Add RSI to the new DataFrame
            momentum_df['rsi'] = ta.momentum.RSIIndicator(
                close=data['close'], window=self.rsi_period).rsi()

        # Add MACD (Moving Average Convergence Divergence)
        if self.macd_short is not None:
            macd = ta.trend.MACD(
                close=data['close'],
                window_slow=self.macd_long,
                window_fast=self.macd_short,
                window_sign=self.macd_signal
            )
            ###### It looks like MACD line need four columns, not economic ########
            momentum_df['macd'] = macd.macd()
            momentum_df['macd_signal'] = macd.macd_signal()
            momentum_df['macd_diff'] = macd.macd_diff()
            momentum_df['macd_long_ema'] = ta.trend.EMAIndicator(close=data['close'], window=self.macd_long).ema_indicator()

        # Add Stochastic Oscillator
        if self.stoch_period is not None:
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
        if self.bollinger_period is not None:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=data['close'], window=self.bollinger_period, window_dev=self.bollinger_std)
            volatility_df['bollinger_mavg'] = bollinger.bollinger_mavg() # Middle band (SMA)
            volatility_df['bollinger_upper'] = bollinger.bollinger_hband()
            volatility_df['bollinger_lower'] = bollinger.bollinger_lband()
            
        if self.atr_period is not None:
            # Average True Range (ATR)
            volatility_df['atr'] = ta.volatility.AverageTrueRange(
                high=data['high'], low=data['low'], close=data['close'], window=self.atr_period
            ).average_true_range()
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
            ).volume_weighted_average_price()
        if self.obv_lookback is not None:
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
        if self.sma_period is not None:
            # Simple Moving Average (SMA)
            trend_df['sma'] = ta.trend.SMAIndicator(
                close=data['close'], window=self.sma_period
            ).sma_indicator()
        if self.ema_period is not None:
            # Exponential Moving Average (EMA)
            trend_df['ema'] = ta.trend.EMAIndicator(
                close=data['close'], window=self.ema_period
            ).ema_indicator()
        if self.adx_period is not None:
            # Average Directional Index (ADX)
            trend_df['adx'] = ta.trend.ADXIndicator(
                high=data['high'], low=data['low'], close=data['close'], window=self.adx_period
            ).adx()
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
            momentum_df = self.add_momentum_indicators(data)
            volatility_df = self.add_volatility_indicators(data)
            volume_df = self.add_volume_indicators(data)
            trend_df = self.add_trend_indicators(data)
            custom_df = self.add_custom_indicators(data)
            self.indicators[symbol] = pd.concat([momentum_df, volatility_df, volume_df, trend_df, custom_df], axis=1)
            del momentum_df, volatility_df, volume_df, trend_df, custom_df
            gc.collect()

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
        else: # no selection method specified
            self.selections = self.indicators[self.symbols[0]].columns.tolist()
        for symbol in self.symbols:
            self.indicators[symbol] = self.indicators[symbol][self.selections]




############## Real-time update functions #################
    def update(self, new_data):

        new_datetime = new_data.pop('datetime')
        for symbol in self.symbols:
            data_symbol = new_data[symbol]
            new_row = pd.DataFrame({col: 0 for col in self.indicators[symbol].columns}, index=[new_datetime])
            self.indicators[symbol] = pd.concat([self.indicators[symbol], new_row])
            self.update_mom_indicators(symbol, new_datetime, data_symbol)
            self.update_vol_indicators(symbol, new_datetime, data_symbol)
            self.update_trend_indicators(symbol, new_datetime, data_symbol)
            self.update_custom_indicators(symbol, new_datetime, data_symbol)


    def update_mom_indicators(self, symbol, new_datetime, data):
        if 'rsi' in self.indicators[symbol].columns:
            self.indicators[symbol].at[new_datetime, 'rsi'] = self.update_rsi(symbol)
        if 'macd' in self.indicators[symbol].columns:
            prev_macd = {
                'short_ema': self.indicators[symbol]['long_ema'].iloc[-2]-self.indicators[symbol]['macd'].iloc[-2],
                'long_ema': self.indicators[symbol]['macd_long_ema'].iloc[-2],
                'macd_signal': self.indicators[symbol]['macd_signal'].iloc[-2]
            }
            new_close = data['close']
            updated_macd = self.update_macd(prev_macd, new_close)
            self.indicators[symbol].at[new_datetime, 'macd'] = updated_macd['macd_value']
            self.indicators[symbol].at[new_datetime, 'macd_signal'] = updated_macd['macd_signal']
            self.indicators[symbol].at[new_datetime, 'macd_diff'] = updated_macd['macd_diff']
            self.indicators[symbol].at[new_datetime, 'macd_long_ema'] = updated_macd['long_ema']

        if 'stoch_k' in self.indicators[symbol].columns and 'stoch_d' in self.indicators[symbol].columns:
            prev_stochastic = {
                'highest_high': self.indicators[symbol]['lookback_high'].iloc[-2],
                'lowest_low': self.indicators[symbol]['lookback_low'].iloc[-2],
                'stoch_d': self.indicators[symbol]['stoch_d'].iloc[-2],
            }
            updated_stochastic = self.update_stochastic(
                prev_stochastic, data
            )
            # Update the momentum_df with the new stoch_k and stoch_d values
            self.indicators[symbol].at[new_datetime, 'stoch_k'] = updated_stochastic['stoch_k']
            self.indicators[symbol].at[new_datetime, 'stoch_d'] = updated_stochastic['stoch_d']
            self.indicators[symbol].at[new_datetime, 'lookback_high'] = updated_stochastic['highest_high']
            self.indicators[symbol].at[new_datetime, 'lookback_low'] = updated_stochastic['lowest_low']

                
    def update_vol_indicators(self, symbol, new_datetime, data):
        """
        Updates volatility indicators (Bollinger Bands, ATR) using the last calculated values and the new price data.
        
        :param symbol: The trading symbol to update.
        :param new_datetime: The timestamp of the new data.
        :param data: The latest data, containing 'close', 'high', 'low'.
        """
        # Update Bollinger Bands
        if 'bollinger_mavg' in self.indicators[symbol].columns:
            prev_bollinger = {
                'sma': self.indicators[symbol]['bollinger_mavg'].iloc[-2], # Middle band (SMA)
                'stddev': (self.indicators[symbol]['bollinger_upper'].iloc[-2] - self.indicators[symbol]['bollinger_mavg'].iloc[-2]) / self.bollinger_std
            }
            new_close = data['close']
            updated_bollinger = self.update_bollinger_bands(prev_bollinger, new_close)
            self.indicators[symbol].at[new_datetime, 'bollinger_mavg'] = updated_bollinger['sma']
            self.indicators[symbol].at[new_datetime, 'bollinger_upper'] = updated_bollinger['upper_band']
            self.indicators[symbol].at[new_datetime, 'bollinger_lower'] = updated_bollinger['lower_band']
        
        # Update ATR
        if 'atr' in self.indicators[symbol].columns:
            prev_atr = {
                'atr': self.indicators[symbol]['atr'].iloc[-2],
                'prev_close': self.indicators[symbol]['close'].iloc[-2]
            }
            updated_atr = self.update_atr(prev_atr, data['high'], data['low'], data['close'])
            self.indicators[symbol].at[new_datetime, 'atr'] = updated_atr['atr']


    def update_trend_indicators(self, symbol, new_datetime, data):
        """
        Updates trend-based indicators (SMA, EMA, ADX) using the last calculated values and the new price data.

        :param symbol: The trading symbol to update.
        :param new_datetime: The timestamp of the new data.
        :param data: The latest data, containing 'close', 'high', 'low'.
        """
        # Update Simple Moving Average (SMA)
        if 'sma' in self.indicators[symbol].columns:
            prev_sma = self.indicators[symbol]['sma'].iloc[-2]
            new_close = data['close']
            updated_sma = self.update_sma(prev_sma, new_close)
            self.indicators[symbol].at[new_datetime, 'sma'] = updated_sma

        # Update Exponential Moving Average (EMA)
        if 'ema' in self.indicators[symbol].columns:
            prev_ema = self.indicators[symbol]['ema'].iloc[-2]
            new_close = data['close']
            updated_ema = self.update_ema(prev_ema, new_close)
            self.indicators[symbol].at[new_datetime, 'ema'] = updated_ema

        # Update Average Directional Index (ADX)
        if 'adx' in self.indicators[symbol].columns:
            prev_adx = self.indicators[symbol]['adx'].iloc[-2]
            prev_data = self.data_handler.get_data_limit(symbol, 2, clean=True)[-2]
            prev_high = prev_data['high']
            prev_low = prev_data['low']
            prev_close = prev_data['close']
            current_high = data['high']
            current_low = data['low']
            updated_adx = self.update_adx(prev_high, prev_low, prev_close, current_high, current_low, prev_adx)
            self.indicators[symbol].at[new_datetime, 'adx'] = updated_adx
    
    def update_custom_indicators(self, symbol, new_datetime):
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
            'avg_loss': avg_loss,
            'previous_close': new_close
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