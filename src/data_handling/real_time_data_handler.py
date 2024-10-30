"""
The class of realtime datahandler
"""
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.data_handler import DataHandler, DataCleaner
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
            formatted_record = self.format(record)
            if self.stream.tell() + len(formatted_record) >= self.maxBytes:
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



## Sliding window class for memory monitoring

class SlidingWindow:
    def __init__(self, max_size):
        """
        Initialize the sliding window with a maximum size.
        :param max_size: Maximum number of elements the window can hold.
        """
        self.max_size = max_size
        self.data = []  # List to store the data

    def add(self, new_data):
        """
        Add new data to the sliding window and remove old data if the window exceeds the max size.
        :param new_data: The new data point to add.
        """
        self.data.append(new_data)
        if len(self.data) > self.max_size:
            # Remove the oldest data (FIFO behavior)
            self.data.pop(0)

    def get_latest_data(self):
        """
        Get the latest data points stored in the window.
        :return: List of the most recent data.
        """
        return self.data

    def clear(self):
        """
        Clear all data from the sliding window.
        """
        self.data = []




# Refactored RealTimeDataHandler class
class RealTimeDataHandler(DataHandler):
    def __init__(self, source_file, input_file=None):
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
        # self.scaler_dir = 'data/scaler'
        self.retry_attempts = self.config_handler.get_config('retry_if_error',5)

        # self.scaler= self.config_handler.get_config('scaler')
        self.log_settings['path'] = os.path.join(os.path.dirname(__file__),'../..', self.log_settings['path'])

        self.data_logger = LoggingHandler(self.log_settings['path'],f"{self.log_settings['file_name1']}_{self.interval_str}.log").logger
        # self.data_logger.info("RealTimeDataHandler initialized with config from: {}".format(input_file))
        self.time_logger = LoggingHandler(self.log_settings['path'],f"{self.log_settings['file_name2']}_{self.interval_str}.log").logger
        self.memory_logger = LoggingHandler(self.log_settings['path'],f"{self.log_settings['file_name3']}_{self.interval_str}.log").logger
        self.warning_logger = LoggingHandler(self.log_settings['path'],f"{self.log_settings['file_name4']}_{self.interval_str}.log").logger
        self.error_logger = LoggingHandler(self.log_settings['path'],f"{self.log_settings['file_name5']}_{self.interval_str}.log").logger
        # Initialize sliding window (fixed-size buffer) for cleaned and rescaled data
        
        self.cleaned_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        # self.rescaled_data = {symbol: pd.DataFrame() for symbol in self.symbols}

        memory_setting = self.config_handler.get_config('memory_setting', {'window_size': 1000, 'memory_limit': 80})
        self.memory_limit = memory_setting['memory_limit'] # Memory limit in percentage
        self.window_size = memory_setting['window_size'] # Sliding window size
        self.current_month = datetime.now().strftime("%Y-%m")
        self.current_week = datetime.now().strftime("%Y-%W")

        self.subscribers = [] # List of subscribers to notify when new data is available

    def append_real_time_data(self, df, symbol, process=False, rescaled=False):
        # Append to the cleaned data dictionary for the respective symbol
        if process:
            if self.cleaned_data[symbol].empty:  # Check if DataFrame is empty
                self.cleaned_data[symbol] = df
            else:
                #df_cleaned = self.clean_data(df)
                self.cleaned_data[symbol] = pd.concat([self.cleaned_data[symbol], df])
        if len(self.cleaned_data[symbol]) > self.window_size:
            # Drop the oldest rows to keep only the last `window_size` rows
            self.cleaned_data[symbol] = self.cleaned_data[symbol].tail(self.window_size)

        """
            if rescaled:
            if self.rescaled_data[symbol].empty:  # Check if DataFrame is empty
                self.rescaled_data[symbol] = df
            else:
                # df_rescaled = self.rescale_data(df, symbol)
                self.rescaled_data[symbol] = pd.concat([self.rescaled_data[symbol], df])
        if len(self.rescaled_data[symbol]) > self.window_size:
            # Drop the oldest rows to keep only the last `window_size` rows
            self.rescaled_data[symbol] = self.rescaled_data[symbol].iloc[-self.window_size:]
                
        """

        
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


    def check_for_new_month(self):
        """
        Check if a new month has started, and if so, start new files and move the month-before-last data to historical.
        """
        new_month = datetime.now().strftime("%Y-%m")
        if new_month != self.current_month:
            self.memory_logger.info(f"New month detected. Archiving previous data and starting a new file for {new_month}.")
            for symbol in self.symbols:
                self.transfer_old_data(symbol)
            # Update the current month
            self.current_month = new_month

    """def check_for_new_week(self):
        
        Check if a new month has started, and if so, start new files and move the month-before-last data to historical.
        
        new_week = datetime.now().strftime("%Y-%W")
        if new_week != self.current_week:
            self.memory_logger.info(f"New week detected. Updating the scalers for {new_week}.")
            self.update_scaler()
            self.current_week = new_week"""

    def check_new_data(self, new_data):
        closure = False
        for symbol in self.symbols:
            if new_data[symbol].empty:
                closure = True
                break
            elif 'volume' in new_data[symbol].columns and new_data[symbol]['volume'].sum() < 1e-6:
                closure = True
                break
        return closure
    
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
            return 0
        self.data_logger.info(f"Fetching missing data from {last_fetch_time} to {current_time}")
        # Loop through the missing time range in intervals and fetch data
        time_cursor = last_fetch_time
        month_tracker = time_cursor.strftime("%Y-%m")
        # print("Fetching missing data started., time_cursor: ", time_cursor)
        while time_cursor < current_time:
            time_cursor += self.interval
            month_tracker = time_cursor.strftime("%Y-%m")
            if time_cursor >= current_time:
                break
            #     time_cursor = current_time  # Ensure not to fetch data beyond the current time
            for symbol in self.symbols:
                try:
                    df_raw = self.get_data_at_time(symbol, time_cursor)
                    # Process and save the fetched data
                    if not df_raw.empty:
                        df_time = df_raw.set_index('open_time')
                        self.save_real_time_data(df_time, symbol, month_tracker, raw=True)
                        self.data_logger.info(f"Fetched and saved raw data for {symbol} at {datetime.now(timezone.utc)}")
                        
                        # Process and save the fetched data
                        df_cleaned = self.clean_data(df_raw)
                        df_time = df_cleaned.set_index('open_time')
                        self.append_real_time_data(df_time, symbol, process=True)
                        self.save_real_time_data(df_time, symbol, month_tracker, process=True)
                        self.data_logger.info(f"Cleaned and saved processed data for {symbol} at {datetime.now(timezone.utc)}")
                        # Rescale and save the processed data
                        """
                        df_rescaled = self.rescale_data(df_cleaned, symbol)
                        df_time = df_rescaled.set_index('open_time')
                        self.append_real_time_data(df_time, symbol, rescaled=True)
                        self.save_real_time_data(df_time, symbol, month_tracker, rescaled=True)
                        self.data_logger.info(f"Rescaled and saved rescaled data for {symbol} at {datetime.now(timezone.utc)}")
                        """
                except Exception as e:
                    self.error_logger.error(f"Error fetching missing data for {symbol}: {e}")
            current_time = datetime.now(timezone.utc)  # Update the current time, sometimes the loop goes beyond the current time
            current_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval
 
        self.data_logger.info("Fetching missing data completed., time_cursor: %s", time_cursor)

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
            self.error_logger.error(f"Error fetching real-time data: {e}")
            return pd.DataFrame()
        return df

        
    def file_path(self, symbol, date, process=False, raw=False, rescaled=False, file_type='csv'):
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month = date_obj.strftime('%Y-%m')
        if raw:
            return f'data/real_time/raw/{symbol}_{self.interval_str}_{month}.{file_type}'
        """if rescaled:
            return f'data/real_time/rescaled/{symbol}_{self.interval_str}_{month}.{file_type}'"""
        if process:
            return f'data/real_time/processed/{symbol}_{self.interval_str}_{month}.{file_type}'
        else:
            self.error_logger.error("Please specify the type of data to save.")

    def file_path_month(self, month, symbol, process=False, raw=False, rescaled=False, file_type='csv'):
        # date is dummy here
        if raw:
            return f'data/real_time/raw/{symbol}_{self.interval_str}_{month}.{file_type}'
        """if rescaled:
            return f'data/real_time/rescaled/{symbol}_{self.interval_str}_{month}.{file_type}'"""
        if process:
            return f'data/real_time/processed/{symbol}_{self.interval_str}_{month}.{file_type}'
        
        else:
            self.error_logger.error("Please specify the type of data to save.")
        
    # def file_path_historical(self, symbol, month, process=False, raw=False, rescaled=False, file_type='csv'):
    #     # date is dummy here
    #     if raw:
    #         return f'data/historical/raw/from_real/{symbol}_{self.interval_str}_{month}.{file_type}'
    #     if rescaled:
    #         return f'data/historical/rescaled/from_real/{symbol}_{self.interval_str}_{month}.{file_type}'
    #     if process:
    #         return f'data/historica/processed/from_real/{symbol}_{self.interval_str}_{month}.{file_type}'
    #     else:
    #         print("Please specify the type of data to save.")


    def flush_data(self, symbol):
        # Flush old data to disk when memory limit is exceeded
        cleaned_data = pd.concat(list(self.cleaned_data[symbol]), ignore_index=True)

        # rescaled_data = pd.concat(list(self.rescaled_data[symbol]), ignore_index=True)

        self.cleaned_data[symbol].clear()
        # self.rescaled_data[symbol].clear()

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


    """def load_scaler_path(self, symbol, column):
        
        Loads the scaler from a pickle file based on the symbol, column, and scaling type.
        :param scaler_dir: Directory where scalers are stored
        :param symbol: The symbol (e.g., ADAUSDT, BTCUSDT)
        :param column: The column for which the scaler is required (e.g., 'open', 'close')
        :param scaling_type: Type of scaler ('minmax' or 'standard'), default is 'minmax'
        :return: The loaded scaler or None if not found
        
        scaler_filename = f"{column}_{self.scaler}.pkl"
        scaler_path = os.path.join(self.scaler_dir, symbol ,self.interval_str, scaler_filename)
        
        return scaler_path"""
    
    def load_last_fetch_time(self):
        """
        Load the last fetch time from the log file by reading the last entry.
        :return: The last fetch time as a datetime object or None if not found.
        """
        log_file_path = os.path.join(self.log_settings['path'], f"{self.log_settings['file_name2']}_{self.interval_str}.log")
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


    def load_initial(self):
        def check_required_labels(df, required_labels):
            missing_labels = [label for label in required_labels if label not in df.columns and 'time' not in label]
            if missing_labels:
                self.error_logger.error(f"Missing required labels {missing_labels}")
                return False
            return True

        last_second_month = (datetime.now().replace(day=1) - pd.DateOffset(months=2)).strftime("%Y-%m")
        last_month = (datetime.now().replace(day=1) - pd.DateOffset(months=1)).strftime("%Y-%m")
        required_labels = self.cleaner_kwargs['required_labels']
        for symbol in self.symbols:
            processed_file_path = self.file_path_month(symbol=symbol, month=last_month, process=True)
            if os.path.exists(processed_file_path):
                df = pd.read_csv(processed_file_path, index_col='open_time', parse_dates=True)
                print(df.index.values)
                input("LastMonth, Press Enter to continue...")
                if check_required_labels(df, required_labels):
                    self.data_logger.info(f"Loading processed data for {symbol} from {processed_file_path}")
                    self.append_real_time_data(df, symbol, process=True)
                else:
                    self.warning_logger.warning(f"Required labels missing in processed data file for {symbol} in {processed_file_path}")
                    # print("check_required_labels failed for processed data")
            else:
                self.warning_logger.warning(f"No processed data file found for {symbol} in {processed_file_path}")

            processed_file_path = self.file_path_month(symbol=symbol, month=self.current_month, process=True)
            if os.path.exists(processed_file_path):
                df = pd.read_csv(processed_file_path, index_col='open_time', parse_dates=True)
                if check_required_labels(df, required_labels):
                    self.data_logger.info(f"Loading processed data for {symbol} from {processed_file_path}")
                    self.append_real_time_data(df, symbol, process=True)
                else:
                    self.warning_logger.warning(f"Required labels missing in processed data file for {symbol} in {processed_file_path}")
                    # print("check_required_labels failed for processed data")
            else:
                self.warning_logger.warning(f"No processed data file found for {symbol} in {processed_file_path}")

        # Load rescaled data files into self.rescaled_data
        """for symbol in self.symbols:
            rescaled_file_path = self.file_path_month(symbol=symbol, month=last_month, rescaled=True)
            if os.path.exists(rescaled_file_path):
                df = pd.read_csv(rescaled_file_path, index_col='open_time', parse_dates=True)
                if check_required_labels(df, required_labels):
                    self.data_logger.info(f"Loading rescaled data for {symbol} from {rescaled_file_path}")
                    self.append_real_time_data(df, symbol, rescaled=True)
                else:
                    self.warning_logger.warning(f"Required labels missing in rescaled data file for {symbol} in {rescaled_file_path}")
                    # print("check_required_labels failed for processed data")
            else:
                self.warning_logger.warning(f"No rescaled data file found for {symbol} in {rescaled_file_path}")

            rescaled_file_path = self.file_path_month(symbol=symbol, month=self.current_month, rescaled=True)
            if os.path.exists(rescaled_file_path):
                df = pd.read_csv(rescaled_file_path, index_col='open_time', parse_dates=True)
                if check_required_labels(df, required_labels):
                    self.data_logger.info(f"Loading rescaled data for {symbol} from {rescaled_file_path}")
                    self.append_real_time_data(df, symbol, rescaled=True)
                else:
                    self.warning_logger.warning(f"Required labels missing in rescaled data file for {symbol} in {rescaled_file_path}")
                    # print("check_required_labels failed for processed data")
            else:
                self.warning_logger.warning(f"No rescaled data file found for {symbol} in {rescaled_file_path}")"""

    """def rescale_data(self, df, symbol):
        
        Rescale the data using the appropriate scaler loaded from the saved pickle files.
        :param df: DataFrame with data to be rescaled
        :param symbol: The symbol (e.g., ADAUSDT, BTCUSDT) for which the data is being rescaled
        :param interval: The interval (e.g., '15m', '1h') to determine which scaler to use
        :return: Rescaled DataFrame
        
        if symbol in self.symbols:
            # Define the path to the scalers based on the symbol and interval
            scaler_dir = os.path.join(self.scaler_dir, symbol, self.interval_str)
            if not os.path.exists(scaler_dir):
                # print("scaler_dir: ", scaler_dir)
                self.warning_logger.warning(f"Scalers not found for {symbol} at interval {self.interval_str}. Skipping rescaling.")
                return df
            # Loop through required labels and apply the corresponding scaler
            for col in self.cleaner_kwargs['required_labels']:
                if col in df.columns and 'time' not in col:
                    scaler = joblib.load(self.load_scaler_path(symbol, col))
                    if scaler:
                        # print(col ,"before rescaling: ", df)
                        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
                        # print(col ,"after rescaling: ", df[col].values)
                    else:
                        self.warning_logger.warning(f"Scaler for {col} not found for {symbol} at interval {self.interval_str}. Skipping rescaling.")
        # print("after: ", df)
        return df"""
    
    def save_real_time_data(self, df, symbol, month=None, process=False, raw=False, rescaled=False):
        if month:
            output_file = self.file_path_month(symbol=symbol, month=month, process=process, raw=raw, rescaled=rescaled)
        else:
            output_file = self.file_path(symbol=symbol, date=datetime.now().strftime('%Y-%m-%d'), process=process, raw=raw, rescaled=rescaled)
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode='w', header=True, index=True)
        else:
            df.to_csv(output_file, mode='a', header=False, index=True)
        self.data_logger.info(f"Real-time data saved to {output_file}")


    def transfer_old_data(self,symbol):
        """
        Transfer data from the month before last for raw, processed, and rescaled files to the historical directory.
        """
        current_month = datetime.now()
        month_before_last = current_month.replace(day=1) - pd.DateOffset(months=2)

        # Transfer each type of data (raw, processed, rescaled)
        success = 1
        while success == 1:
            success = self.transfer_to_history(symbol=symbol, month=month_before_last.strftime("%Y-%m"))
            month_before_last = month_before_last.replace(day=1) - pd.DateOffset(months=1)

        
    def transfer_to_history(self, symbol, month):
        """
        Transfer data files from the month before last to the historical directory.
        This applies to raw, processed, and rescaled data files.
        :param symbol: The trading symbol (e.g., 'BTCUSDT').
        """
        count = 0
        # Transfer raw, processed, and rescaled files
        for data_type in ['raw', 'processed', 'rescaled']:
            if data_type == 'raw':
                file_to_transfer = self.file_path_month(symbol=symbol, month=month, raw=True)
            elif data_type == 'processed':
                file_to_transfer = self.file_path_month(symbol=symbol, month=month, process=True)
            """elif data_type == 'rescaled':
                file_to_transfer = self.file_path_month(symbol=symbol, month=month, rescaled=True)"""

            # Check if the file exists and transfer to historical directory
            if os.path.exists(file_to_transfer):
                historical_dir = f'data/historical/{data_type}/from_real'
                os.makedirs(historical_dir, exist_ok=True)  # Create directory if it doesn't exist
                shutil.move(file_to_transfer, os.path.join(historical_dir, os.path.basename(file_to_transfer)))
                self.memory_logger.info(f"Transferred {file_to_transfer} to {historical_dir}")
            else:
                # print(file_to_transfer)
                self.memory_logger.warning(f"No {data_type} file found for {month}, skipping transfer.")
                count += 1
        if count == 0:
            return 1 # Success
        else:
            return 0 # some file skipped
        
    """def update_scaler(self):
        
        Fetch the most recent data, update the existing scaler, and save the updated scaler.

        :param scaler_type: The type of scaler ('minmax' or 'standard').
        :param scaler_save_base_path: Base path where the scalers are saved, organized by symbol and interval.
        :param recent_data_path: Path where the recent data will be fetched from.
        :param update_frequency: Number of data points to use for fitting the scaler (default is 672 for weekly updates).
            # Fetch recent data from the API or CSV
            # if recent_data_path is None:
            #     recent_data = self.fetch_klines_with_limit(symbol, self.interval_str, update_frequency)
            # else:
            #     recent_data_file = os.path.join(recent_data_path, f"{symbol}_{self.interval_str}.csv")
            #     if not os.path.exists(recent_data_file):
            #         self.data_logger.warning(f"Recent data for {symbol} at {self.interval_str} not found. Fetching new data.")
            #         recent_data = self.fetch_klines_with_limit(symbol, self.interval_str, update_frequency)
            #     else:
            #         recent_data = pd.read_csv(recent_data_file)
        
        # recent_data_path = os.path.join('data/real_time/processed')
        one_week = timedelta(weeks=1)
        update_frequency = one_week // self.interval
        for symbol in self.symbols:
            self.data_logger.info(f"Processing symbol: {symbol}")

            
            recent_data = self.fetch_klines_with_limit(symbol, self.interval_str, update_frequency)
            # Define the path where the scaler models for the symbol and interval are saved
            scaler_save_path = os.path.join(self.scaler_dir, symbol, self.interval_str)

            # Ensure the save path exists
            os.makedirs(scaler_save_path, exist_ok=True)

            # Load the existing scaler models or initialize new scalers
            scalers = {}
            for label in self.cleaner_kwargs['required_labels']:
                scaler_file = os.path.join(scaler_save_path, f"{label}_{self.scaler}.pkl")
                if os.path.exists(scaler_file):
                    scalers[label] = joblib.load(scaler_file)
                    self.data_logger.info(f"Loaded existing scaler for {symbol} {label}.")
                else:
                    # Initialize a new scaler if none exists
                    scalers[label] = MinMaxScaler() if self.scaler == 'minmax' else StandardScaler()
                    self.data_logger.info(f"No existing scaler found for {symbol} {label}. Initializing a new scaler.")

            # Fit the scaler on the most recent data (overriding the old one)
            for label in self.cleaner_kwargs['required_labels']:
                if label in recent_data.columns:
                    scaler = scalers[label]
                    scaler.fit(recent_data[label].values.reshape(-1, 1))
                    # Save the updated scaler model (overriding the existing one)
                    joblib.dump(scaler, os.path.join(scaler_save_path, f"{label}_{self.scaler}.pkl"))
                    self.data_logger.info(f"Updated and saved scaler for {symbol} {label}.")

            self.data_logger.info(f"Scalers updated for {symbol} with {self.interval_str} data.")
    """
    def data_fetch_loop(self, next_fetch_time, last_fetch_time):
        new_data = {}
        self.check_for_new_month()
        #self.check_for_new_week()
        for symbol in self.symbols:
            if next_fetch_time <= last_fetch_time:
                self.data_logger.info(f"Skipping fetch for {symbol} at {next_fetch_time}")
                continue
            df_raw = self.get_data_at_time(symbol, next_fetch_time)
            df_time = df_raw.set_index('open_time')
            self.save_real_time_data(df_time, symbol, raw=True)
            self.data_logger.info(f"Fetched and saved raw data for {symbol} at {datetime.now(timezone.utc)}")

            df_cleaned = self.clean_data(df_raw)
            df_time = df_cleaned.set_index('open_time')
            self.append_real_time_data(df_time, symbol, process=True)  
            self.save_real_time_data(df_time, symbol, process=True)
            self.data_logger.info(f"Cleaned and saved processed data for {symbol} at {datetime.now(timezone.utc)}")
            new_data[symbol] = df_time
            #print(symbol, df_time)
            """df_rescaled = self.rescale_data(df_cleaned, symbol)
            df_time = df_rescaled.set_index('open_time')
            self.append_real_time_data(df_time, symbol, rescaled=True)
            self.save_real_time_data(df_time, symbol, rescaled=True)
            self.data_logger.info(f"Rescaled and saved rescaled data for {symbol} at {datetime.now(timezone.utc)}")
            """

        if 'df_raw' in locals():
            del df_raw
        if 'df_cleaned' in locals():
            del df_cleaned
        if 'df_rescaled' in locals():
            del df_rescaled
        if 'df_time' in locals():
            del df_time
        gc.collect()
        # Save last fetch time and log it
        fetch_time = datetime.now(timezone.utc)
        fetch_time = fetch_time.replace(second=0, microsecond=0)
        self.time_logger.info(f"Last fetch time: {next_fetch_time}")
        self.data_logger.info(f"Saved last fetch time: {next_fetch_time}")
        # At the end of the loop for each symbol
        return new_data
        
        
    def pre_run_data(self):
        """
        Prepares the environment before running the main process.
        Loads initial data, determines the last fetch time, and calculates the next fetch time.
        
        Returns:
            tuple: (next_fetch_time, last_fetch_time)
        """
        self.load_initial()  # Load some initial data in the memory
        last_fetch_time = None
        log_file_path = os.path.join(self.log_settings['path'],f"{self.log_settings['file_name2']}_{self.interval_str}.log")
        
        if os.path.exists(log_file_path):
            last_fetch_time = self.load_last_fetch_time()
            

        
        current_time = datetime.now(timezone.utc)
        if last_fetch_time is None:  # If no last fetch time is found, fetch data for the last 30 intervals
            last_fetch_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval - self.window_size * self.interval
        
        if last_fetch_time <= current_time - self.interval:
            self.fetch_missing_data(last_fetch_time)
        current_time = datetime.now(timezone.utc) # Update the current time after fetching missing data
        next_fetch_time = current_time - (current_time - datetime.min.replace(tzinfo=timezone.utc)) % self.interval
        last_fetch_month = last_fetch_time.strftime("%Y-%m")

        if last_fetch_month != self.current_month:
            pass

        return next_fetch_time, last_fetch_time
        
    def sample_loop(self, next_fetch_time, last_fetch_time):
        new_data = self.data_fetch_loop(next_fetch_time, last_fetch_time)
        self.notify_subscribers(new_data)
        del new_data
        gc.collect()
        now = datetime.now(timezone.utc)
        next_fetch_time = self.calculate_next_grid(now)
        sleep_duration = (next_fetch_time - now).total_seconds()
        self.data_logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
        time.sleep(sleep_duration)
        return next_fetch_time



    def sample_run(self):
        next_fetch_time,last_fetch_time = self.pre_run_data()
        while True:
            '''
            # (Optional) is not used for crypto data
            # if not self.is_market_open():
            #     self.logger.info("Market is closed. Skipping data fetch.")
            #     time.sleep(60)
            #     continue
            '''
            self.data_fetch_loop(next_fetch_time, last_fetch_time)
            now = datetime.now(timezone.utc)
            next_fetch_time = self.calculate_next_grid(now)
            sleep_duration = (next_fetch_time - now).total_seconds()
            self.data_logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")

            time.sleep(sleep_duration)


######### For Subscribers #########

    def subscribe(self, subscriber):
        """Add a new subscriber to the list."""
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)
    def unsubscribe(self, subscriber):
        """Remove a subscriber from the list."""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

    def notify_subscribers(self, new_data):
        for subscriber in self.subscribers:
            subscriber.update(new_data)

    def get_data(self, symbol, clean=False, rescale=False):
        if clean:
            return self.cleaned_data[symbol]
        """if rescale:
            return self.rescaled_data[symbol]"""
        
    def get_last_data(self, symbol, clean=False, rescale=False):
        if clean:
            return self.cleaned_data[symbol].iloc[-1]
        """if rescale:
            return self.rescaled_data[symbol].iloc[-1]"""
        
    def get_data_limit(self, symbol, limit, clean=False, rescale=False):
        if clean:
            return self.cleaned_data[symbol].tail(limit)
        """if rescale:
            return self.rescaled_data[symbol].tail(limit)"""
        

    """
    Each subscriber (e.g., Feature, SignalProcessing, Model) will register with RealTimeDataHandler, 
    allowing the data handler to notify them when new data is available. 
    This avoids constant polling and unnecessary overhead in RealtimeDealer.
    
    def notify_subscribers(self, new_data):
        for subscriber in self.subscribers:
            subscriber.update(new_data)
    """


if __name__ == '__main__':
    
    # Initialize the RealTimeDataHandler
    data_handler = RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')
    data_handler.sample_run()


# # Create test data
    # test_data = {
    #     "timestamp": ["2024-07-01 00:00:00", "2024-07-01 01:00:00"],
    #     "price": [100, 101],
    #     "volume": [1000, 1050]
    # }

    # # Create a DataFrame
    # df = pd.DataFrame(test_data)

    # # Save files for 'BTCUSDT' symbol in 'raw', 'processed', and 'rescaled' for July 2024
    # file_paths = [
    #     'data/real_time/raw/BTCUSDT_1m_2024-07.csv',
    #     'data/real_time/processed/BTCUSDT_1m_2024-07.csv',
    #     'data/real_time/rescaled/BTCUSDT_1m_2024-07.csv',
    #     'data/real_time/raw/BTCUSDT_1m_2024-06.csv',
    #     'data/real_time/processed/BTCUSDT_1m_2024-06.csv',
    #     'data/real_time/rescaled/BTCUSDT_1m_2024-06.csv'
    # ]

    # # Write the test data to the files
    # for file_path in file_paths:
    #     df.to_csv(file_path, index=False)

    # # List the created files for verification
    # created_files = os.listdir('data/real_time/raw') + os.listdir('data/real_time/processed') + os.listdir('data/real_time/rescaled')
    