
"""
The class of historical datahandler
"""
from data_handler import DataHandler
import requests
import pandas as pd
import pytz
from datetime import datetime as dt
import time


from data_handler import DataHandler

class HistoricalDataHandler(DataHandler):
    def __init__(self, source, frequency):
        """
        Initializes the historical data handler with source details and frequency.
        :param source: Dictionary containing 'base_url' and 'endpoint'.
        :param frequency: Data frequency (e.g., '1d', '1h').
        """
        super().__init__(source, frequency)


    def ensure_correct_format(self, date_str):
        try:
            # Try to parse the date string
            date_obj = dt.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If parsing fails, reformat the date string
            date_obj = pd.to_datetime(date_str, utc=True)
            date_str = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return date_str
    
    
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
        start_time = int(dt.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000)
        end_time = int(dt.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000) if end_date else None

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
    

    def fetch_klines_with_limit(self, symbol, interval, limit):
        """
        Get historical klines (candlestick data) for a given symbol and interval with a specified limit.
        It is for small data requests (e.g., 500 records).
        """
        url = self.build_url()

        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(data, columns=columns)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        return df

    def fetch_historical_small(self, symbol, interval, start_date, end_date=None):
        """
        Get historical klines (candlestick data) for a given symbol and interval.
        This function is used for small data requests (e.g., 500 records).
        """
        
        
        start_date = self.ensure_correct_format(start_date)
        end_date = self.ensure_correct_format(end_date) if end_date else None
        url = self.build_url()

        start_time = int(dt.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000)
        end_time = int(dt.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).timestamp() * 1000) if end_date else None
        
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

    def clean_data(self, data):
        # Code for cleaning historical data
        pass

    def rescale_data(self, data):
        # Optional: Code for rescaling data
        pass
