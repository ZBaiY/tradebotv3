"""
The class of realtime datahandler
"""
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.data_handler import DataHandler, ScalerHandler, DataCleaner, DataChecker, rescale_data
import pytz
import logging
from datetime import datetime, timedelta, timezone

import json
import shutil
import logging
import pandas as pd
import re
import requests
import numpy as np
import time
import gc
import psutil
import pickle
from collections import deque
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

### Logging file rolling handler
class SizeLimitedFileHandler(logging.FileHandler):
    def __init__(self, filename, maxBytes=1024*100, backupCount=1, encoding=None, delay=False):
        super().__init__(filename, mode='a', encoding=encoding, delay=delay)
        self.maxBytes = maxBytes
        self.backupCount = backupCount

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            if self.should_rollover(record):
                self.do_rollover()
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)

    def should_rollover(self, record):
        """
        Determine if rollover should occur.
        """
        if self.stream is None:  # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            self.stream.seek(0, 2)  # due to non-posix-compliant Windows feature
            if self.stream.tell() + len(self.format(record)) >= self.maxBytes:
                return True
        return False

    def do_rollover(self):
        """
        Do a rollover, as in RotatingFileHandler but without creating new files.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Read the current log file
        with open(self.baseFilename, 'r') as f:
            lines = f.readlines()
        # Remove the oldest entries
        current_size = os.path.getsize(self.baseFilename)
        while len(lines) > 0 and current_size > self.maxBytes:
            current_size -= len(lines.pop(0))

        with open(self.baseFilename, 'w') as f:        # Write the remaining lines back to the log file
            f.writelines(lines)
        self.stream = self._open()# Reopen the stream


# Helper class to handle logging
class LoggingHandler:
    def __init__(self, log_dir='../../data/real_time/logs', log_file='data_logs.log'):
        self.log_file = os.path.join(os.path.dirname(__file__), log_dir, log_file)
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        logger_name = os.path.splitext(log_file)[0]
        self.logger = self.setup_logging(logger_name)

    def setup_logging(self, logger_name):
        logger = logging.getLogger(logger_name)  # Unique logger name
        logger.setLevel(logging.DEBUG)

        # Check if the logger already has handlers (prevents adding multiple handlers)
        if not logger.hasHandlers():
            handler = SizeLimitedFileHandler(self.log_file, maxBytes=1024 * 100, backupCount=1)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False  # Prevent duplicate log messages
        
        return logger  # Return the logger instance
    
# Helper class to load and manage configuration
class ConfigHandler:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            return json.load(file)

    def get_config(self, key, default=None):
        return self.config.get(key, default)


# Refactored RealTimeDataHandler class
class RealTimeDataHandler(DataHandler):
    def __init__(self, source_file, input_file=None, window_size=1000, memory_limit=50):
        super().__init__(source_file)

        # Initialize the logger
        if input_file is None:
            input_file = os.path.join(os.path.dirname(__file__), 'config/fetch_real_time.json')
        self.config_handler = ConfigHandler(input_file)

        self.log_settings = self.config_handler.get_config('log_setting', {})
        self.symbols = self.config_handler.get_config('symbols')
        self.interval_str = self.config_handler.get_config('interval')
        unit_mapping = {
            'm': 'minutes','h': 'hours', 'd': 'days','s': 'seconds'
        }
        self.interval_unit = self.interval_str[-1]  # Extract the unit (e.g., 'm' for minutes)
        self.interval_value = int(self.interval_str[:-1])  # Extract the numeric part and convert to integer
        if self.interval_unit in unit_mapping:
            self.interval = timedelta(**{unit_mapping[self.interval_unit]: self.interval_value})
        else:
            raise ValueError(f"Unsupported interval unit: {self.interval_unit}")
        self.filetype = self.config_handler.get_config('filetype')
        self.cleaner_kwargs = {}
        self.cleaner_kwargs['params'] = self.config_handler.get_config('params')
        self.cleaner_kwargs['required_labels'] = self.config_handler.get_config('required_labels')
        self.base_url = self.build_url()
        self.scaler_dir = '../../data/scaler'
        self.retry_attempts = self.config_handler.get_config('retry_if_error',5)

        self.scaler= self.config_handler.get_config('scaler')
        self.log_settings['path'] = os.path.join(os.path.dirname(__file__),'../..', self.log_settings['path'])

        self.data_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name1']).logger
        # self.data_logger.info("RealTimeDataHandler initialized with config from: {}".format(input_file))
        self.time_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name2']).logger
        self.memory_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name3']).logger
        self.warning_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name4']).logger
        self.error_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name5']).logger
        # Initialize sliding window (fixed-size buffer) for cleaned and rescaled data
        self.window_size = window_size
        self.cleaned_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.rescaled_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.memory_limit = memory_limit # Memory limit in percentage


    def append_real_time_data(self, df, symbol, process=False, rescaled=False):
        # Append to the cleaned data dictionary for the respective symbol
        if process:
            if self.cleaned_data[symbol].empty:  # Check if DataFrame is empty
                self.cleaned_data[symbol] = df
            else:
                df_cleaned = self.clean_data(df)
                self.cleaned_data[symbol] = pd.concat([self.cleaned_data[symbol], df_cleaned], ignore_index=True)
        
        if rescaled:
            if self.rescaled_data[symbol].empty:  # Check if DataFrame is empty
                self.rescaled_data[symbol] = df
            else:
                df_rescaled = self.rescale_data(df, symbol)
                self.rescaled_data[symbol] = pd.concat([self.rescaled_data[symbol], df_rescaled], ignore_index=True)
                
    def build_url_with_time_range(self, symbol, start_time, end_time):
        """
        Builds a URL for fetching data with a specific time range for a symbol.
        :param symbol: The symbol to fetch data for.
        :param start_time: The start of the time range.
        :param end_time: The end of the time range.
        :return: The URL for the API request.
        """
        # Convert time to the correct format for the API (assumed format: UNIX timestamp)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        url = f"{self.base_url}?symbol={symbol}&interval={self.interval_str}&startTime={start_timestamp}&endTime={end_timestamp}"
        return url
    
        
    def calculate_next_grid(self, current_time):
        next_grid_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval + self.interval
        if next_grid_time <= current_time:
            #next_grid_time += self.interval
            self.error_logger.error("Next grid time is less than current time")
        return next_grid_time
    
    def clean_data(self, df):
        cleaner = DataCleaner(df, **self.cleaner_kwargs)
        return cleaner.get_cleaned_df()

    def check_memory_limit(self):
        # Check system memory usage and return True if it exceeds the defined limit
        memory_usage = psutil.virtual_memory().percent
        return memory_usage > self.memory_limit



                
    def fetch_missing_data(self, last_fetch_time):
        """
        Fetches missing data between the last fetch time and now.
        :param last_fetch_time: The last fetch time as a datetime object.
        """
        current_time = datetime.now(timezone.utc)
        current_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval
        missing_time_range = current_time - last_fetch_time

        # Determine the interval for fetching (e.g., 15-minute intervals)
        if missing_time_range < self.interval:
            self.data_logger.info("No missing data to fetch.")
            return
        self.data_logger.info(f"Fetching missing data from {last_fetch_time} to {current_time}")

        # Loop through the missing time range in intervals and fetch data
        time_cursor = last_fetch_time
        print("Fetching missing data started., time_cursor: ", time_cursor)
        while time_cursor < current_time:
            time_cursor += self.interval
            if time_cursor >= current_time:
                break
            #     time_cursor = current_time  # Ensure not to fetch data beyond the current time

            for symbol in self.symbols:
                try:
                    df_raw = self.get_data_at_time(symbol, time_cursor)
                    # Process and save the fetched data
                    if not df_raw.empty:
                        self.save_real_time_data(df_raw, symbol, raw=True)
                        self.data_logger.info(f"Fetched and saved raw data for {symbol} at {datetime.now(timezone.utc)}")

                        # Process and save the fetched data
                        df_cleaned = self.clean_data(df_raw)
                        self.append_real_time_data(df_cleaned, symbol, process=True)
                        self.save_real_time_data(df_cleaned, symbol, process=True)
                        self.data_logger.info(f"Cleaned and saved processed data for {symbol} at {datetime.now(timezone.utc)}")

                        # Rescale and save the processed data
                        df_rescaled = self.rescale_data(df_cleaned, symbol)
                        self.append_real_time_data(df_rescaled, symbol, rescaled=True)
                        self.save_real_time_data(df_rescaled, symbol, rescaled=True)
                        self.data_logger.info(f"Rescaled and saved rescaled data for {symbol} at {datetime.now(timezone.utc)}")
                except Exception as e:
                    self.error_logger.error(f"Error fetching missing data for {symbol}: {e}")
        
        print("Fetching missing data completed., time_cursor: ", time_cursor)


    def fetch_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching real-time data: {e}")
            return pd.DataFrame()
        return df

        
    def file_path(self, symbol, date, process=False, raw=False, rescaled=False, file_type='csv'):

        if raw:
            return f'data/real_time/raw/{symbol}_{self.interval_str}_{date}.{file_type}'
        if rescaled:
            return f'data/real_time/rescaled/{symbol}_{self.interval_str}_{date}.{file_type}'
        if process:
            return f'data/real_time/processed/{symbol}_{self.interval_str}_{date}.{file_type}'
        else:
            print("Please specify the type of data to save.")


    def flush_data(self, symbol):
        # Flush old data to disk when memory limit is exceeded
        cleaned_data = pd.concat(list(self.cleaned_data[symbol]), ignore_index=True)
        rescaled_data = pd.concat(list(self.rescaled_data[symbol]), ignore_index=True)

        self.cleaned_data[symbol].clear()
        self.rescaled_data[symbol].clear()

        # Run garbage collection after clearing old data
        gc.collect()

    def get_data_at_time(self, symbol, target_time):
        # Calculate the start and end time in milliseconds
        start_time = int(target_time.timestamp() * 1000)
        end_time = int((target_time + self.interval-timedelta(seconds=1)).timestamp() * 1000)
        
        # Construct the URL with startTime and endTime
        url = f"{self.base_url}?symbol={symbol}&interval={self.interval_str}&startTime={start_time}&endTime={end_time}"
        df_raw = self.fetch_data(url)
        return df_raw



    def get_cleaned_data(self, symbol):
        """
        Also attach some historical data to the cleaned data
        """

        pass

    def load_scaler(self, symbol, column):
        """
        Loads the scaler from a pickle file based on the symbol, column, and scaling type.
        :param scaler_dir: Directory where scalers are stored
        :param symbol: The symbol (e.g., ADAUSDT, BTCUSDT)
        :param column: The column for which the scaler is required (e.g., 'open', 'close')
        :param scaling_type: Type of scaler ('minmax' or 'standard'), default is 'minmax'
        :return: The loaded scaler or None if not found
        """
        scaler_filename = f"{symbol}_{column}_{self.scaler}.pkl"
        scaler_path = os.path.join(self.scaler_dir, scaler_filename)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
                self.data_logger.info(f"Loaded {self.scaler} scaler for {symbol} - {column} from {scaler_path}")
                return scaler
        else:
            self.warning_logger.warning(f"Scaler file {scaler_path} not found.")
            return None
    

    def load_last_fetch_time(self):
        """
        Load the last fetch time from the log file by reading the last entry.
        :return: The last fetch time as a datetime object or None if not found.
        """
        log_file_path = os.path.join(self.log_settings['path'], self.log_settings['file_name2'])
        last_fetch_time = None

        # Open the log file and read it in reverse to find the last fetch time entry
        try:
            with open(log_file_path, 'r') as log_file:
                lines = log_file.readlines()

            # Traverse the log file in reverse order to find the last fetch time
            for line in reversed(lines):
                match = re.search(r'Last fetch time: (.+)', line)
                if match:
                    last_fetch_time_str = match.group(1).strip()
                    # Use '%Y-%m-%d %H:%M:%S%z' to handle timezone information
                    last_fetch_time = datetime.strptime(last_fetch_time_str, '%Y-%m-%d %H:%M:%S%z')
                    break

        except FileNotFoundError:
            self.warning_logger.warning(f"Log file {log_file_path} not found. No previous fetch time available.")
        except Exception as e:
            self.error_logger.error(f"Error reading log file: {e}")

        if last_fetch_time:
            self.data_logger.info(f"Last fetch time loaded: {last_fetch_time}")
        else:
            self.data_logger.info("No previous fetch time found in the log.")

        return last_fetch_time
    

    def load_params(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}


    def rescale_data(self, df, symbol):
        """
        Rescale the data using the appropriate scaler loaded from the saved pickle files.
        :param df: DataFrame with data to be rescaled
        :param symbol: The symbol (e.g., ADAUSDT, BTCUSDT) for which the data is being rescaled
        :param interval: The interval (e.g., '15m', '1h') to determine which scaler to use
        :return: Rescaled DataFrame
        """
        if symbol in self.symbols:
            # Define the path to the scalers based on the symbol and interval
            scaler_dir = os.path.join(self.scaler_dir, symbol, self.interval_str)
            if not os.path.exists(scaler_dir):
                self.warning_logger.warning(f"Scalers not found for {symbol} at interval {self.interval_str}. Skipping rescaling.")
                return df
            # Loop through required labels and apply the corresponding scaler
            for col in self.cleaner_kwargs['required_labels']:
                if col in df.columns and 'time' not in col:
                    scaler = self.load_scaler(scaler_dir, symbol, col)
                    if scaler:
                        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
                    else:
                        self.warning_logger.warning(f"Scaler for {col} not found for {symbol} at interval {self.interval_str}. Skipping rescaling.")
        
        return df
    
    def save_real_time_data(self, df, symbol, process=False, raw=False, rescaled=False):
        output_file = self.file_path(symbol=symbol, date=datetime.now().strftime('%Y-%m-%d'), process=process, raw=raw, rescaled=rescaled)
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode='w', header=True, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)
        self.data_logger.info(f"Real-time data saved to {output_file}")


    def save_last_fetch_time(self, fetch_time):
        # Append the last fetch time to the log file
        with open(self.log_settings['path'], 'a') as log_file:
            log_file.write(f"Last fetch time for {self.symbols}: {fetch_time}\n")

    def run(self):

        last_fetch_time = None
        if os.path.exists(os.path.join(self.log_settings['path'],self.log_settings['file_name2'])):
            last_fetch_time = self.load_last_fetch_time()

        current_time = datetime.now(timezone.utc)
        if last_fetch_time is None: # If no last fetch time is found, fetch data for the last 30 intervals
            last_fetch_time =  current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval- 10*self.interval
        
        if last_fetch_time <= current_time - self.interval:
            self.fetch_missing_data(last_fetch_time)
        
        next_fetch_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval
        
        while True:
            '''
            # (Optional) is not used for crypto data
            # if not self.is_market_open():
            #     self.logger.info("Market is closed. Skipping data fetch.")
            #     time.sleep(60)
            #     continue
            '''
            for symbol in self.symbols:
                if next_fetch_time <= last_fetch_time:
                    self.data_logger.info(f"Skipping fetch for {symbol} at {next_fetch_time}")
                    continue
                df_raw = self.get_data_at_time(symbol, next_fetch_time)
                self.save_real_time_data(df_raw, symbol, raw=True)
                self.data_logger.info(f"Fetched and saved raw data for {symbol} at {datetime.now(timezone.utc)}")

                df_cleaned = self.clean_data(df_raw)
                self.append_real_time_data(df_cleaned, symbol, process=True)  
                self.save_real_time_data(df_cleaned, symbol, process=True)
                self.data_logger.info(f"Cleaned and saved processed data for {symbol} at {datetime.now(timezone.utc)}")

                df_rescaled = self.rescale_data(df_cleaned, symbol)
                self.append_real_time_data(df_rescaled, symbol, rescaled=True)
                self.save_real_time_data(df_rescaled, symbol, rescaled=True)
                self.data_logger.info(f"Rescaled and saved rescaled data for {symbol} at {datetime.now(timezone.utc)}")
    
            if 'df_raw' in locals():
                del df_raw
            if 'df_cleaned' in locals():
                del df_cleaned
            if 'df_rescaled' in locals():
                del df_rescaled
            gc.collect()
            # Save last fetch time and log it
            fetch_time = datetime.now(timezone.utc)
            fetch_time = fetch_time.replace(second=0, microsecond=0)
            self.time_logger.info(f"Last fetch time: {fetch_time}")
            self.data_logger.info(f"Saved last fetch time: {fetch_time}")
            # At the end of the loop for each symbol
            
            now = datetime.now(timezone.utc)
            next_fetch_time = self.calculate_next_grid(now)
            sleep_duration = (next_fetch_time - now).total_seconds() + 5
            self.data_logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
            time.sleep(sleep_duration)


if __name__ == '__main__':
    data_handler = RealTimeDataHandler(source_file='config/source.json', input_file='config/fetch_real_time.json')
    data_handler.run()