
"""
The class of historical datahandler
"""

import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.data_handler import DataHandler, DataCleaner, DataChecker, rescale_data
import json
import requests
import pandas as pd
import pytz
import joblib
from datetime import datetime 
import time
from tqdm import tqdm


class HistoricalDataHandler(DataHandler):
    def __init__(self, source_file = 'config/source.json', json_file=None, cleaner_file=None, checker_file=None):
        """
        Initializes the historical data handler with source details, frequency, and optional JSON parameter files.
        :param source: Dictionary containing 'base_url' and 'endpoint'.
        :param frequency: Data frequency (e.g., '1d', '1h').
        :param cleaner_param_file: Path to JSON file containing data cleaning parameters.
        :param checker_param_file: Path to JSON file containing data check parameters.
        """
        super().__init__(source_file)
        if cleaner_file is None:
            cleaner_file = os.path.join(os.path.dirname(__file__), 'config/cleaner.json')
        if checker_file is None:
            checker_file = os.path.join(os.path.dirname(__file__), 'config/checker.json')
        self.cleaner_params = self.load_params(cleaner_file)
        self.checker_params = self.load_params(checker_file)
        with open(json_file, 'r') as file:
            fetch_config = json.load(file)

        self.symbols = [symbol_config['symbol'] for symbol_config in fetch_config['symbols']]
        self.cleaned_data = {symbol: pd.DataFrame() for symbol in self.symbols}

########################### Functions for loading data and backtesting ########################################

    def load_data(self, interval_str='15m', begin_date='2023-01-01', end_date='2024-09-24'):
        base_path = 'data/historical/processed/for_train/'
        for symbol in self.symbols:
            file_path = f'{base_path}{symbol}_{begin_date}_{end_date}_{interval_str}.csv'
            self.cleaned_data[symbol] = pd.read_csv(file_path)

        self.window_size = len(self.cleaned_data[self.symbols[0]])
        self.interval_str = interval_str

        return self.cleaned_data

    def get_data(self, symbol, clean=True, rescale=False):
        if clean:
            return self.cleaned_data[symbol]
        """if rescale:
            return self.rescaled_data[symbol]"""
        
    def get_last_data(self, symbol, clean=True, rescale=False):
        if clean:
            return self.cleaned_data[symbol].iloc[-1]
        """if rescale:
            return self.rescaled_data[symbol].iloc[-1]"""
        
    def get_data_limit(self, symbol, limit, clean=True, rescale=False):
        if clean:
            return self.cleaned_data[symbol].tail(limit)
        """if rescale:
            return self.rescaled_data[symbol].tail(limit)"""
        
    def copy(self):
        copied = HistoricalDataHandler()
        copied.cleaned_data = {symbol: data.copy() for symbol, data in self.cleaned_data.items()}
        return copied
    
########################### Functions for fetching data ########################################

    def load_params(self, file_path):
        """
        Loads parameters from a JSON file.
        :param file_path: Path to the JSON file.
        :return: Dictionary containing the parameters.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}
    

    def ensure_correct_format(self, date_str):
        try:
            # Try to parse the date string
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If parsing fails, reformat the date string
            date_obj = pd.to_datetime(date_str, utc=True)
            date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return date_str
    
    def file_path(self, symbol, interval, start_date, end_date=None, process=False, raw=False, rescaled=False, file_type='csv'):
        if end_date is None:
            end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        if raw:
            return f'data/historical/for_train/raw/{symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if rescaled:
            return f'data/historical/for_train/rescaled/{symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if process:
            return f'data/historical/for_train/processed/{symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        else:
            print("Please specify the type of data to save.")
    
    
    def fetch_data_chunk(self, symbol, interval, start_date, end_date=None, limit=500, rate_limit_delay=1):
        """
        Fetches a chunk of historical kline data for a symbol in a given interval.
        :param symbol: Symbol (e.g., BTCUSDT).
        :param interval: Kline interval (e.g., '1d', '1h', '1min').
        :param start_date: Start date (string format).
        :param end_date: End date (optional).
        :param limit: Max number of records to retrieve (default 500).
        :param rate_limit_delay: Delay between retries on failure (default 1 second).
        :return: DataFrame containing kline data.
        """
        # Ensure dates are in the correct format
        start_date = self.ensure_correct_format(start_date)
        end_date = self.ensure_correct_format(end_date) if end_date else None

        # Convert dates to timestamps in milliseconds
        start_time = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000)
        end_time = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000) if end_date else None

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'limit': limit
        }

        if end_time:
            params['endTime'] = end_time

        url = self.build_url()

        while True:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore'
                ])

                # Convert timestamps
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

                return df

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}. Retrying in {rate_limit_delay} seconds...")
                time.sleep(rate_limit_delay)
                continue
    

    def fetch_historical_small(self, symbol, interval, start_date, end_date=None):
        """
        Get historical klines (candlestick data) for a given symbol and interval.
        This function is used for small data requests (e.g., 500 records).
        """
        
        
        start_date = self.ensure_correct_format(start_date)
        end_date = self.ensure_correct_format(end_date) if end_date else None
        url = self.build_url()

        start_time = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000)
        end_time = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000) if end_date else None
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(data, columns=columns)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        return df

        
    def save_data_chunks(self, symbol, interval, start_date, end_date=None, limit=500, rate_limit_delay=1, file_type='csv'):
        """
        Fetches data in chunks, cleans it, performs checks, and saves it to CSV or HDF5.
        :param symbol: Symbol (e.g., 'BTCUSDT').
        :param interval: Data frequency (e.g., '1d', '1h').
        :param start_date: Start date.
        :param end_date: End date (optional).
        :param limit: Max number of records per request.
        :param rate_limit_delay: Delay between API requests to avoid rate limits.
        :param file_type: File type to save the data ('csv' or 'h5').
        """
        current_start_date = start_date
        first_chunk = True
        output_file_path = self.file_path(symbol, interval, start_date, end_date, process=True, file_type=file_type)
        end_date = end_date or datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        interval_value = int(''.join(filter(str.isdigit, interval)))
        interval_unit = ''.join(filter(str.isalpha, interval))

        # For tqdm progress bar
        if interval_unit == 'd':
            total_duration = ((end_dt - start_dt).days)*interval_value
            unit = "days"
            period_default = f'{interval_value}D'
        elif interval_unit == 'h':
            total_duration = ((end_dt - start_dt).days * 24 + (end_dt - start_dt).seconds // 3600)//interval_value
            unit = "hours"
            period_default = f'{interval_value}h'
        elif interval_unit == 'm':
            total_duration = ((end_dt - start_dt).days * 24 * 60 + (end_dt - start_dt).seconds // 60)//interval_value
            unit = "minutes"
            period_default = f'{interval_value}min'
        elif interval_unit == 's':
            total_duration = ((end_dt - start_dt).days * 24 * 60 * 60 + (end_dt - start_dt).seconds)//interval_value
            unit = "seconds"
            period_default = f'{interval_value}S'
        else:
            raise ValueError("Unsupported interval format. Use 's', 'm', 'h', or 'd'.")


        pbar = tqdm(total=total_duration, desc="Downloading data", unit=unit)

        while True:
            df_chunk = self.fetch_data_chunk(symbol, interval, current_start_date, end_date, limit, rate_limit_delay)
            if df_chunk.empty:
                break
            
            # Data cleaning using DataCleaner with loaded JSON parameters
            data_cleaner = DataCleaner(df_chunk, **self.cleaner_params)
            cleaned_df = data_cleaner.get_cleaned_df()

            # Data checking using DataChecker with loaded JSON parameters
            data_checker = DataChecker(cleaned_df, self.checker_params['check_params'], self.checker_params.get('expected_types'))
            results = data_checker.perform_check()
            cleaned_df = cleaned_df.set_index('open_time')
            if not results['is_clean']:
                if file_type == 'csv':
                    cleaned_df.to_csv(output_file_path, mode='a', header=True, index=True)
                elif file_type == 'h5':
                    cleaned_df.to_hdf(output_file_path, key='df', mode='w' if first_chunk else 'a', format='table', append=not first_chunk)
                print("Data chunk is not clean, please check the results.")
                print(results)
                return None

            # Save the cleaned chunk
            if file_type == 'csv':
                cleaned_df.to_csv(output_file_path, mode='w' if first_chunk else 'a', header=first_chunk, index=True)
            elif file_type == 'h5':
                cleaned_df.to_hdf(output_file_path, key='df', mode='w' if first_chunk else 'a', format='table', append=not first_chunk)
            
            first_chunk = False

            pbar.update(len(cleaned_df))
            current_start_date = (pd.to_datetime(cleaned_df['open_time'].iloc[-1], utc=True) + pd.Timedelta(**{unit: interval_value})).strftime('%Y-%m-%d %H:%M:%S')
            if pd.to_datetime(current_start_date, utc=True) >= end_dt:
                break

        pbar.close()
        print(f"Data is clean and saved to {output_file_path}")
        return None

    def save_raw_chunks(self, symbol, interval, start_date, end_date=None, limit=500, rate_limit_delay=1, file_type = 'csv'):
        """
        Save raw historical data chunks without cleaning or checking.
        :param symbol: Trading symbol (e.g., BTCUSDT).
        :param interval: Data interval (e.g., '1d', '1h').
        :param start_date: Start date.
        :param end_date: End date (optional).
        :param output_file: File path to save the raw data.
        :param limit: Max number of records per request.
        :param rate_limit_delay: Delay between retries to avoid API rate limits.
        """
        current_start_date = start_date
        first_chunk = True
        output_file_path = self.file_path(symbol, interval, start_date, end_date, raw=True, file_type=file_type)
        end_date = end_date or datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        start_dt, end_dt = pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True)
        interval_value = int(''.join(filter(str.isdigit, interval)))
        interval_unit = ''.join(filter(str.isalpha, interval))

        # For tqdm progress bar
        if interval_unit == 'd':
            total_duration = ((end_dt - start_dt).days)*interval_value
            unit = "days"
            period_default = f'{interval_value}D'
        elif interval_unit == 'h':
            total_duration = ((end_dt - start_dt).days * 24 + (end_dt - start_dt).seconds // 3600)//interval_value
            unit = "hours"
            period_default = f'{interval_value}h'
        elif interval_unit == 'm':
            total_duration = ((end_dt - start_dt).days * 24 * 60 + (end_dt - start_dt).seconds // 60)//interval_value
            unit = "minutes"
            period_default = f'{interval_value}min'
        elif interval_unit == 's':
            total_duration = ((end_dt - start_dt).days * 24 * 60 * 60 + (end_dt - start_dt).seconds)//interval_value
            unit = "seconds"
            period_default = f'{interval_value}S'
        else:
            raise ValueError("Unsupported interval format. Use 's', 'm', 'h', or 'd'.")

        pbar = tqdm(total=total_duration, desc="Downloading raw data", unit=unit)

        while True:
            df_chunk = self.fetch_data_chunk(symbol, interval, current_start_date, end_date, limit, rate_limit_delay)
            if df_chunk.empty: break
            df_chunk = df_chunk.set_index('open_time')
            # Save raw data chunk
            if file_type == 'csv':
                df_chunk.to_csv(output_file_path, mode='w' if first_chunk else 'a', header=first_chunk, index=True)
            elif file_type == 'h5':
                df_chunk.to_hdf(output_file_path, key='df', mode='w' if first_chunk else 'a', format='table', append=not first_chunk)
            
            first_chunk = False

            pbar.update(len(df_chunk))
            current_start_date = (pd.to_datetime(df_chunk['open_time'].iloc[-1], utc=True) + pd.Timedelta(**{unit: interval_value})).strftime('%Y-%m-%d %H:%M:%S')
            if pd.to_datetime(current_start_date, utc=True) >= end_dt:
                break

        pbar.close()
        print(f"Raw data is saved to {output_file_path}")
    
    def save_rescaled_chunks(self, symbol, interval, start_date, end_date=None, scaler='minmax', limit=500, rate_limit_delay=1, file_type = 'csv'):
        """
        Fetches data in chunks, cleans it, resamples it, rescales it, and saves it to CSV.
        :param symbol: Symbol (e.g., 'BTCUSDT').
        :param interval: Data frequency (e.g., '1d', '1h').
        :param start_date: Start date.
        :param end_date: End date (optional).
        :param limit: Max number of records per request.
        :param rate_limit_delay: Delay between API requests to avoid rate limits.
        """
        current_start_date = start_date
        first_chunk = True
        output_file_path = self.file_path(symbol, interval, start_date, end_date, rescaled=True, file_type=file_type)
        end_date = end_date or datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        interval_value = int(''.join(filter(str.isdigit, interval)))
        interval_unit = ''.join(filter(str.isalpha, interval))

        # For tqdm progress bar
        if interval_unit == 'd':
            total_duration = ((end_dt - start_dt).days)*interval_value
            unit = "days"
            period_default = f'{interval_value}D'
        elif interval_unit == 'h':
            total_duration = ((end_dt - start_dt).days * 24 + (end_dt - start_dt).seconds // 3600)//interval_value
            unit = "hours"
            period_default = f'{interval_value}h'
        elif interval_unit == 'm':
            total_duration = ((end_dt - start_dt).days * 24 * 60 + (end_dt - start_dt).seconds // 60)//interval_value
            unit = "minutes"
            period_default = f'{interval_value}min'
        elif interval_unit == 's':
            total_duration = ((end_dt - start_dt).days * 24 * 60 * 60 + (end_dt - start_dt).seconds)//interval_value
            unit = "seconds"
            period_default = f'{interval_value}S'
        else:
            raise ValueError("Unsupported interval format. Use 's', 'm', 'h', or 'd'.")

        pbar = tqdm(total=total_duration, desc="Downloading and rescaling data", unit=unit)

        while True:
            df_chunk = self.fetch_data_chunk(symbol, interval, current_start_date, end_date, limit, rate_limit_delay)
            if df_chunk.empty:
                break

            # Data cleaning using DataCleaner with loaded JSON parameters
            data_cleaner = DataCleaner(df_chunk, **self.cleaner_params)
            cleaned_df = data_cleaner.get_cleaned_df()

            # Rescale the cleaned DataFrame
            
            rescaled_df = rescale_data(cleaned_df, scaler)
            rescaled_df = rescaled_df.set_index('open_time')
            # Data checking using DataChecker with loaded JSON parameters, the rescaler can produce unlogical values, so only check the cleaned data
            # data_checker = DataChecker(rescaled_df, self.checker_params['check_params'], self.checker_params.get('expected_types'))
            # results = data_checker.perform_check()

            # Save the rescaled chunk
            if file_type == 'csv':
                rescaled_df.to_csv(output_file_path, mode='w' if first_chunk else 'a', header=first_chunk, index=True)
            elif file_type == 'h5':
                rescaled_df.to_hdf(output_file_path, key='df', mode='w' if first_chunk else 'a', format='table', append=not first_chunk)
            
            first_chunk = False

            pbar.update(len(rescaled_df))
            current_start_date = (pd.to_datetime(rescaled_df['open_time'].iloc[-1], utc=True) +
                                pd.Timedelta(**{unit: int(interval[:-1])})).strftime('%Y-%m-%d %H:%M:%S')

            if pd.to_datetime(current_start_date, utc=True) >= end_dt:
                break

        pbar.close()
        print(f"Rescaled data is clean and saved to {output_file_path}")
        return None

    def fetch_save_json(self, json_file):
        """
        Fetches data and saves it based on the configuration in the provided JSON file.
        :param json_file: Path to the JSON file containing the fetch configuration.
        """
        with open(json_file, 'r') as file:
            fetch_config = json.load(file)

        limit = fetch_config.get('limit', 500)
        rate_limit_delay = fetch_config.get('rate_limit_delay', 1)
        file_type = fetch_config.get('file_type', 'csv')
        scaler = fetch_config.get('scaler', 'standard')

        for symbol_config in fetch_config['symbols']:
            symbol = symbol_config['symbol']
            for interval_config in symbol_config['intervals']:
                interval = interval_config['interval']
                start_date = interval_config['start_date']
                end_date = interval_config.get('end_date')

                raw = interval_config.get('raw', False)
                rescale = interval_config.get('rescale', False)
                clean = interval_config.get('clean', False)

                if raw:
                    # Save raw chunks without cleaning or rescaling
                    print(f"Fetching raw data for {symbol} at interval {interval}")
                    self.save_raw_chunks(symbol, interval, start_date, end_date, limit=limit, rate_limit_delay=rate_limit_delay, file_type=file_type)
                if clean:
                    # Save cleaned data chunks
                    print(f"Fetching cleaned data for {symbol} at interval {interval}")
                    self.save_data_chunks(symbol, interval, start_date, end_date, limit=limit, rate_limit_delay=rate_limit_delay, file_type=file_type)
                if rescale:
                    # Save rescaled data chunks
                    print(f"Fetching rescaled data for {symbol} at interval {interval}")
                    self.save_rescaled_chunks(symbol, interval, start_date, end_date, scaler = scaler, limit=limit, rate_limit_delay=rate_limit_delay, file_type=file_type)
                


class SingleSymbolDataHandler:
    def __init__(self, symbol, source_file = 'config/source.json', json_file=None, cleaner_file=None, checker_file=None):
        """
        Initializes the single symbol data handler with source details, frequency, and optional JSON parameter files.
        :param symbol: The trading symbol (e.g., 'BTCUSD').
        :param source_file: Path to the data source.
        :param json_file: Path to the JSON file containing configuration details.
        :param cleaner_file: Path to JSON file containing data cleaning parameters.
        :param checker_file: Path to JSON file containing data check parameters.
        """
        self.symbol = symbol
        self.source_file = source_file
        self.json_file = json_file or 'backtest/config/data_h.json'
        if cleaner_file is None:
            cleaner_file = os.path.join(os.path.dirname(__file__), 'config/cleaner.json')
        if checker_file is None:
            checker_file = os.path.join(os.path.dirname(__file__), 'config/checker.json')
        
        self.cleaner_params = self.load_params(cleaner_file)
        self.checker_params = self.load_params(checker_file)
        #### missing a lot parameters here, check the real-time data handler for more details
        if self.json_file:
            with open(self.json_file, 'r') as file:
                fetch_config = json.load(file)
            self.config = fetch_config
        else:
            self.config = {}
        self.interval_str = self.config.get('interval', '15m')
        memory_setting = self.config.get('memory_setting', {'window_size': 1000, 'memory_limit': 80})
        self.memory_limit = memory_setting['memory_limit'] # Memory limit in percentage
        self.window_size = memory_setting['window_size'] # Sliding window size
        self.current_month = datetime.now().strftime("%Y-%m")
        self.current_week = datetime.now().strftime("%Y-%W")
        
        self.cleaned_data = pd.DataFrame()
        self.begin_date = None
        self.end_date = None

    ########################### Functions for loading data and backtesting ########################################
    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, interval_str='15m', begin_date='2023-01-01', end_date='2024-09-24'):
        """
        Loads historical data for the specified symbol.
        :param interval_str: Data interval (e.g., '1d', '15m').
        :param begin_date: Start date for the data.
        :param end_date: End date for the data.
        """
        base_path = 'data/historical/processed/for_train/'
        file_path = f'{base_path}{self.symbol}_{begin_date}_{end_date}_{interval_str}.csv'
        self.cleaned_data = pd.read_csv(file_path, index_col='open_time')
        
        # self.window_size = len(self.cleaned_data)
        self.interval_str = interval_str

        return self.cleaned_data

    def get_data(self, clean=True, rescale=False):
        
        if clean:
            return self.cleaned_data

    def get_last_data(self, clean=True, rescale=False):
       
        if clean:
            return self.cleaned_data.iloc[-1]

    def get_data_limit(self, limit, clean=True, rescale=False):
        
        if clean:
            return self.cleaned_data.tail(limit)
        
    def get_data_range(self, start_index, end_index, clean=True, rescale=False):
        if clean:
            return self.cleaned_data.iloc[start_index:end_index]
    def copy(self):
        copied = SingleSymbolDataHandler(self.symbol, self.source_file)
        copied.set_dates(self.start_date, self.end_date)
        copied.cleaned_data = self.cleaned_data.copy()
        return copied
    ########################### Functions for fetching data ########################################

    def load_params(self, file_path):
        """
        Loads parameters from a JSON file.
        :param file_path: Path to the JSON file.
        :return: Dictionary containing the parameters.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}

    def ensure_correct_format(self, date_str):
        """
        Ensures the date string is in the correct format.
        :param date_str: Date string to check and format.
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            date_obj = pd.to_datetime(date_str, utc=True)
            date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return date_str

    def file_path(self, interval, start_date, end_date=None, process=False, raw=False, rescaled=False, file_type='csv'):
        
        if end_date is None:
            end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        if raw:
            return f'data/historical/for_train/raw/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if rescaled:
            return f'data/historical/for_train/rescaled/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if process:
            return f'data/historical/for_train/processed/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        else:
            print("Please specify the type of data to save.")




class MultiSymbolDataHandler:
    def __init__(self, symbols, source_file='config/source.json', json_file=None, cleaner_file=None, checker_file=None):
        """
        Initializes the multi-symbol data handler with source details, frequency, and optional JSON parameter files.
        :param symbols: List of trading symbols (e.g., ['BTCUSD', 'ETHUSD']).
        :param source_file: Path to the data source.
        :param json_file: Path to the JSON file containing configuration details.
        :param cleaner_file: Path to JSON file containing data cleaning parameters.
        :param checker_file: Path to JSON file containing data check parameters.
        """
        self.symbols = symbols
        self.source_file = source_file
        self.json_file = json_file or 'backtest/config/data_h.json'
        if cleaner_file is None:
            cleaner_file = os.path.join(os.path.dirname(__file__), 'config/cleaner.json')
        if checker_file is None:
            checker_file = os.path.join(os.path.dirname(__file__), 'config/checker.json')
        
        self.cleaner_params = self.load_params(cleaner_file)
        self.checker_params = self.load_params(checker_file)
        #### missing a lot parameters here, check the real-time data handler for more details
        
        if self.json_file:
            with open(self.json_file, 'r') as file:
                self.config = json.load(file)
        else:
            self.config = {}

        self.interval_str = self.config.get('interval', '15m')
        memory_setting = self.config.get('memory_setting', {'window_size': 1000, 'memory_limit': 80})
        self.memory_limit = memory_setting['memory_limit'] # Memory limit in percentage
        self.window_size = memory_setting['window_size'] # Sliding window size
        self.current_month = datetime.now().strftime("%Y-%m")
        self.current_week = datetime.now().strftime("%Y-%W")

        self.cleaned_data = {symbol: pd.DataFrame() for symbol in self.symbols}
        # for multi-symbol operations, pack everything into a dictionary
        # There are lots of modules to symphony
        self.subscribers = []

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, interval_str, begin_date, end_date):
        """
        Loads historical data for all symbols.
        :param interval_str: Data interval (e.g., '1d', '15m').
        :param begin_date: Start date for the data.
        :param end_date: End date for the data.
        :return: Dictionary of dataframes keyed by symbol.
        """
        for symbol in self.symbols:
            base_path = 'data/historical/processed/for_train/'
            file_path = f'{base_path}{symbol}_{begin_date}_{end_date}_{interval_str}.csv'
            self.cleaned_data[symbol] = pd.read_csv(file_path, index_col='open_time')

    def get_symbol_data(self, symbol, clean=True, rescale=False):
        """
        Fetches data for a specific symbol.
        :param symbol: Trading symbol (e.g., 'BTCUSD').
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: DataFrame containing the data.
        """
        if symbol in self.symbol_handlers:
            return self.symbol_handlers[symbol].get_data(clean, rescale)
        raise ValueError(f"Symbol {symbol} not managed by this handler.")

    def get_symbol_last_data(self, symbol, clean=True, rescale=False):
        """
        Fetches the last data point for a specific symbol.
        :param symbol: Trading symbol (e.g., 'BTCUSD').
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: Last data point as a Series or row.
        """
        if symbol in self.symbol_handlers:
            return self.symbol_handlers[symbol].get_last_data(clean, rescale)
        raise ValueError(f"Symbol {symbol} not managed by this handler.")
    
    def get_symbol_data_limit(self, symbol, limit, clean=True, rescale=False):
        """
        Fetches the last N rows of data for a specific symbol.
        :param symbol: Trading symbol (e.g., 'BTCUSD').
        :param limit: Number of rows to fetch.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: DataFrame containing the last N rows.
        """
        if symbol in self.symbol_handlers:
            return self.symbol_handlers[symbol].get_data_limit(limit, clean, rescale)
        raise ValueError(f"Symbol {symbol} not managed by this handler.")

    def get_symbol_data_range(self, symbol, start_index, end_index, clean=True, rescale=False):
        """
        Fetches a range of data for a specific symbol.
        :param symbol: Trading symbol (e.g., 'BTCUSD').
        :param start_index: Starting index for the range.
        :param end_index: Ending index for the range.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: DataFrame containing the range of data.
        """
        if symbol in self.symbol_handlers:
            return self.symbol_handlers[symbol].get_data_range(start_index, end_index, clean, rescale)
        raise ValueError(f"Symbol {symbol} not managed by this handler.")

    def get_data(self, clean=True, rescale=False):
        """
        Fetches data for all symbols.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: Dictionary of dataframes keyed by symbol.
        """
        all_data = {}
        for symbol, handler in self.symbol_handlers.items():
            all_data[symbol] = handler.get_data(clean, rescale)
        return all_data

    def get_last_data(self, clean=True, rescale=False):
        """
        Fetches the last data point for all symbols.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: Dictionary of the last data points keyed by symbol.
        """
        all_last_data = {}
        for symbol, handler in self.symbol_handlers.items():
            all_last_data[symbol] = handler.get_last_data(clean, rescale)
        return all_last_data

    def get_data_limit(self, limit, clean=True, rescale=False):
        """
        Fetches the last N rows of data for all symbols.
        :param limit: Number of rows to fetch for each symbol.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: Dictionary of dataframes with the last N rows keyed by symbol.
        """
        all_data_limit = {}
        for symbol, handler in self.symbol_handlers.items():
            all_data_limit[symbol] = handler.get_data_limit(limit, clean, rescale)
        return all_data_limit

    def get_data_range(self, start_index, end_index, clean=True, rescale=False):
        """
        Fetches a range of data for all symbols.
        :param start_index: Starting index for the range.
        :param end_index: Ending index for the range.
        :param clean: Whether to return cleaned data.
        :param rescale: Whether to rescale data.
        :return: Dictionary of dataframes for the range keyed by symbol.
        """
        all_data_range = {}
        for symbol, handler in self.symbol_handlers.items():
            all_data_range[symbol] = handler.get_data_range(start_index, end_index, clean, rescale)
        return all_data_range

    def add_symbol(self, symbol, json_file=None, cleaner_file=None, checker_file=None):
        """
        Adds a new symbol to the handler.
        :param symbol: Trading symbol to add.
        :param json_file: Optional JSON file with symbol-specific configuration.
        :param cleaner_file: Optional JSON file with data cleaning parameters.
        :param checker_file: Optional JSON file with data check parameters.
        """
        if symbol not in self.symbol_handlers:
            self.symbol_handlers[symbol] = SingleSymbolDataHandler(
                symbol, self.source_file, json_file, cleaner_file, checker_file
            )
        else:
            print(f"Symbol {symbol} is already managed.")

    def remove_symbol(self, symbol):
        """
        Removes a symbol from the handler.
        :param symbol: Trading symbol to remove.
        """
        if symbol in self.symbol_handlers:
            del self.symbol_handlers[symbol]
        else:
            print(f"Symbol {symbol} is not managed by this handler.")

    def copy(self):
        copied = MultiSymbolDataHandler(self.symbols, self.source_file)
        copied.cleaned_data = self.cleaned_data.copy()
        return copied
    
    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
    
    def notify_subscribers(self):
        # Notify all subscribers
        # Make sure the order: feature, signal processor -> 
        for subscriber in self.subscribers:
            subscriber.update(self)



    def load_params(self, file_path):
        """
        Loads parameters from a JSON file.
        :param file_path: Path to the JSON file.
        :return: Dictionary containing the parameters.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}

    def ensure_correct_format(self, date_str):
        """
        Ensures the date string is in the correct format.
        :param date_str: Date string to check and format.
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            date_obj = pd.to_datetime(date_str, utc=True)
            date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return date_str

    def file_path(self, interval, start_date, end_date=None, process=False, raw=False, rescaled=False, file_type='csv'):
        
        if end_date is None:
            end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
        if raw:
            return f'data/historical/for_train/raw/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if rescaled:
            return f'data/historical/for_train/rescaled/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        if process:
            return f'data/historical/for_train/processed/{self.symbol}_{start_date}_{end_date}_{interval}.{file_type}'
        else:
            print("Please specify the type of data to save.")
