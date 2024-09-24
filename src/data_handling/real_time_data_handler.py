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
from datetime import datetime, timedelta
import json
import shutil
import logging
import pandas as pd
import re
import requests
import numpy as np
import time
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from logging.handlers import RotatingFileHandler


# Helper class to handle logging
class LoggingHandler:
    def __init__(self, log_file='data_logs.log', log_dir='../../data/real_time'):
        self.log_file = os.path.join(os.path.dirname(__file__), log_dir, log_file)
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        self.logger = self.setup_logging()  # Ensure logger is set as an attribute

    def setup_logging(self):
        logger = logging.getLogger('RealTimeDataHandler')
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(self.log_file, maxBytes=1024 * 100, backupCount=1)

        if os.path.getsize(self.log_file) == 0:
            handler.doRollover()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Logger initialized")
        
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
    def __init__(self, source_file, input_file=None):
        super().__init__(source_file)

        # Initialize the logger
        if input_file is None:
            input_file = os.path.join(os.path.dirname(__file__), 'config/fetch_real_time.json')
        self.config_handler = ConfigHandler(input_file)

        self.log_settings = self.config_handler.get_config('log_setting', {})
        self.symbols = self.config_handler.get_config('symbols')
        self.interval = self.config_handler.get_config('interval')
        self.filetype = self.config_handler.get_config('filetype')
        self.clean_params = self.config_handler.get_config('clean_params')
        self.clean_params['required_labels'] = self.config_handler.get_config('required_labels')
        self.base_url = self.build_url()

        self.scaler_handler = ScalerHandler(self.config_handler.get_config('scalers', {}))

        self.data_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name1']).logger
        self.data_logger.info("RealTimeDataHandler initialized with config from: {}".format(input_file))
        self.time_logger = LoggingHandler(self.log_settings['path'],self.log_settings['file_name2']).logger
        self.cleaned_data = {symbol: pd.DataFrame() for symbol in self.symbols} # Initialize an empty DataFrame to store cleaned data

    def rescale_data(self, df, symbol):
        if symbol in self.symbol:
            for col in self.required_labels:
                if col in df.columns and 'time' not in col:
                    scaler = self.scaler_handler.scalers[symbol].get(col)
                    if scaler:
                        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
        return df

    def load_params(self, file_path):
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}

    def run(self):
        
        
        if os.path.exists(self.log_settings['path']+self.log_settings['file_name2']):
            last_fetch_time = self.load_last_fetch_time()

        if last_fetch_time:
            self.fetch_missing_data(last_fetch_time)

        next_fetch_time = datetime.utcnow()//self.interval * self.interval
        while True:
            # (Optional) is not used for crypto data
            # if not self.is_market_open():
            #     self.logger.info("Market is closed. Skipping data fetch.")
            #     time.sleep(60)
            #     continue
            for symbol in self.symbols:
                df_raw = self.get_data_at_time(symbol, next_fetch_time)
                self.save_real_time_data(df_raw, symbol, raw=True)
                self.data_logger.info(f"Fetched and saved raw data for {symbol} at {datetime.utcnow()}")

                df_cleaned = self.clean_data(df_raw)
                self.append_real_time_data(df_cleaned, symbol)
                self.save_real_time_data(df_cleaned, symbol, process=True)
                self.data_logger.info(f"Cleaned and saved processed data for {symbol} at {datetime.utcnow()}")

                df_rescaled = self.rescale_data(df_cleaned, symbol)
                self.save_real_time_data(df_rescaled, symbol, rescaled=True)
                self.data_logger.info(f"Rescaled and saved rescaled data for {symbol} at {datetime.utcnow()}")

            # Save last fetch time and log it
            fetch_time = datetime.utcnow()
            fetch_time = fetch_time.replace(second=0, microsecond=0)
            self.time_logger.info(f"Last fetch time: {fetch_time}")
            self.data_logger.info(f"Saved last fetch time: {fetch_time}")

            now = datetime.utcnow()
            next_fetch_time = self.calculate_next_grid(now)
            sleep_duration = (next_fetch_time - now).total_seconds() + 5
            self.data_logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
            time.sleep(sleep_duration)

    def load_last_fetch_time(self):
        """
        Load the last fetch time from the log file by reading the last entry.
        :return: The last fetch time as a datetime object or None if not found.
        """
        log_file_path = self.log_settings['path']+self.log_settings['file_name2']
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
                    last_fetch_time = datetime.strptime(last_fetch_time_str, '%Y-%m-%d %H:%M:%S.%f')
                    break

        except FileNotFoundError:
            self.data_logger.warning(f"Log file {log_file_path} not found. No previous fetch time available.")
        except Exception as e:
            self.data_logger.error(f"Error reading log file: {e}")

        if last_fetch_time:
            self.data_logger.info(f"Last fetch time loaded: {last_fetch_time}")
        else:
            self.data_logger.info("No previous fetch time found in the log.")

        return last_fetch_time
    
    def fetch_missing_data(self, last_fetch_time):
        """
        Fetches missing data between the last fetch time and now.
        :param last_fetch_time: The last fetch time as a datetime object.
        """
        current_time = datetime.utcnow()
        current_time = (current_time.minute // self.interval) * self.interval
        missing_time_range = current_time - last_fetch_time

        # Determine the interval for fetching (e.g., 15-minute intervals)
        fetch_interval = timedelta(minutes=self.interval)

        self.data_logger.info(f"Fetching missing data from {last_fetch_time} to {current_time}")

        # Loop through the missing time range in intervals and fetch data
        time_cursor = last_fetch_time
        while time_cursor < current_time:
            time_cursor += fetch_interval
            if time_cursor > current_time:
                time_cursor = current_time  # Ensure not to fetch data beyond the current time

            for symbol in self.symbols:
                try:
                    # Build the API request URL with a time range
                    url = self.build_url_with_time_range(symbol, time_cursor - fetch_interval, time_cursor)

                    # Fetch the data for the given time range
                    df_raw = self.fetch_data(url)

                    # Process and save the fetched data
                    if not df_raw.empty:
                        df_cleaned = self.clean_data(df_raw)
                        self.append_real_time_data(df_cleaned, symbol)
                        self.save_real_time_data(df_cleaned, symbol, process=True)
                        self.data_logger.info(f"Fetched missing data for {symbol} from {time_cursor - fetch_interval} to {time_cursor}")

                except Exception as e:
                    self.data_logger.error(f"Error fetching missing data for {symbol}: {e}")

    def get_data_at_time(self, symbol, target_time):
        # Calculate the start and end time in milliseconds
        start_time = int(target_time.timestamp() * 1000)
        end_time = int((target_time + self.interval-timedelta(minutes=1)).timestamp() * 1000)
        
        # Construct the URL with startTime and endTime
        url = f"{self.base_url}?symbol={symbol}&interval={self.interval}&startTime={start_time}&endTime={end_time}"
        df_raw = self.fetch_data(url)
        return df_raw

    def build_url_with_time_range(self, symbol, start_time, end_time):
        """
        Builds a URL for fetching data with a specific time range for a symbol.
        :param symbol: The symbol to fetch data for.
        :param start_time: The start of the time range.
        :param end_time: The end of the time range.
        :return: The URL for the API request.
        """
        base_url = self.build_url()
        # Convert time to the correct format for the API (assumed format: UNIX timestamp)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        url = f"{base_url}?symbol={symbol}&interval={self.interval}&startTime={start_timestamp}&endTime={end_timestamp}"
        return url

    def calculate_next_grid(self, current_time):
        next_minute = (current_time.minute // self.interval) * self.interval + self.interval
        if next_minute == 60:
            next_fetch_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_fetch_time = current_time.replace(minute=next_minute, second=0, microsecond=0)
        return next_fetch_time

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

    def clean_data(self, df):
        cleaner = DataCleaner(df, **self.clean_params)
        return cleaner.get_cleaned_df()

    def save_real_time_data(self, df, symbol, process=False, raw=False, rescaled=False):
        output_file = self.file_path(symbol=symbol, start_date=datetime.now().strftime('%Y-%m-%d'), process=process, raw=raw, rescaled=rescaled)
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode='w', header=True, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)
        self.data_logger.info(f"Real-time data saved to {output_file}")
        
    def file_path(self, symbol, process=False, raw=False, rescaled=False, file_type='csv'):
        if end_date is None:
            end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        if raw:
            return f'data/historical/raw/{symbol}_{self.interval}.{file_type}'
        if rescaled:
            return f'data/historical/rescaled/{symbol}_{self.interval}.{file_type}'
        if process:
            return f'data/historical/processed/{symbol}_{self.interval}.{file_type}'
        else:
            print("Please specify the type of data to save.")

    def append_real_time_data(self, df, symbol):
        # Append to the cleaned data dictionary for the respective symbol
        if self.cleaned_data[symbol].empty:
            self.cleaned_data[symbol] = df
        else:
            df_cleaned = self.clean_data(df)
            self.cleaned_data[symbol] = pd.concat([self.cleaned_data[symbol], df_cleaned], ignore_index=True)

    def save_last_fetch_time(self, fetch_time):
        # Append the last fetch time to the log file
        with open(self.log_settings['path'], 'a') as log_file:
            log_file.write(f"Last fetch time for {self.symbols}: {fetch_time}\n")


    def get_cleaned_data(self, symbol):
        """
        Also attach some historical data to the cleaned data
        """

        pass

import random
import string
if __name__ == '__main__':
    logging_handler = LoggingHandler()
    logger = logging_handler.logger
    print(f"Log file path: {logging_handler.log_file}")

    # Generate random messages
    for _ in range(10):
        random_message = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        logger.debug(f"Random debug message: {random_message}")
        logger.info(f"Random info message: {random_message}")
        logger.warning(f"Random warning message: {random_message}")
        logger.error(f"Random error message: {random_message}")
        logger.critical(f"Random critical message: {random_message}")
    with open(logging_handler.log_file, 'r') as log_file:
        print(log_file.read())
    # Flush and close all handlers
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    # Open and read the log file
    try:
        with open(logging_handler.log_file, 'r') as log_file:
            print(log_file.read())
    except FileNotFoundError:
        print(f"Log file not found: {logging_handler.log_file}")