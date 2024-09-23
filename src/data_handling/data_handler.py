"""
Base class for data handling. This class should be inherited by other classes that will handle data from different sources.
"""
import json
import pandas as pd
import requests
import time
import numpy as np
import pytz
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataHandler:
    def __init__(self, source_file, frequency = '1h'): # frequency will be used for the loading of the data
        """
        Base class for data handling (historical and real-time).
        :param source: Dictionary containing base_url and other API-related parameters.
        :param frequency: Data frequency (e.g., '1d', '1h', etc.)
        """
        with open(source_file, 'r') as file:
            source = json.load(file)
        
        self.base_url = source.get("base_url")
        self.endpoint = source.get("endpoint")
        self.frequency = frequency


    def build_url(self, additional_params=None):
        """
        Constructs the full URL for API requests, using base_url, endpoint, and optional parameters.
        :param additional_params: Additional URL parameters (if any).
        :return: Complete URL string.
        """
        url = self.base_url + self.endpoint
        if additional_params:
            url += "?" + "&".join([f"{key}={value}" for key, value in additional_params.items()])
        return url
    

    def fetch_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_data(self, data, file_path):
        # Code to save data to file
        pass
    
    
    def load_data(self, file_path):
        # Code to load data from file
        pass
        

def rescale_data(df, scaler_type='standard'):
    """
    Rescale numeric columns in the DataFrame using the specified scaler.
    
    :param df: pd.DataFrame, the DataFrame to process
    :param scaler_type: str, the type of scaler to use ('standard' or 'minmax')
    :return: pd.DataFrame, the DataFrame with rescaled numeric columns
    """
    exclude_columns = ['open_time', 'close_time']
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=exclude_columns, errors='ignore')
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Fit and transform the numeric columns
    scaled_values = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_df.columns, index=numeric_df.index)
    
    # Replace the original numeric columns with the scaled values
    df[numeric_df.columns] = scaled_df
    
    return df

class DataCleaner:
    def __init__(self, df, **kwargs): # params, **kwargs access through the json file
        """
        Base class for data cleaning.
        :param df: DataFrame containing raw data.
        :param params: Dictionary containing various parameters for data processing.
        :param kwargs: Additional keyword arguments, including:
                    - required_labels: List of required labels for the DataFrame.
                    - datetime_format: Datetime format string.
        """

        self.required_labels = kwargs.get('required_labels', [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        # Do the truncation at the beginning
        self.df = df.reindex(columns=self.required_labels, fill_value=pd.NaT if 'open_time' in self.required_labels else 0)
        self.datetime_format = kwargs.get('datetime_format', 'ms')
        self.params = kwargs.get('params', {
            "check_labels": True,
            "dtype": True,
            "resample_align": True,
            "timezone_adjust": False,
            "zero_variance": True,
            "remove_outliers": True,
            "resample_freq": "H", 
            "outlier_threshold": 20, 
            "adjacent_count": 7, 
            "utc_offset": 3
        })
    

    def datatype_df(self):
        """
        Clean missing values, convert data types, and truncate the DataFrame to contain only the required labels.
        """
        # Check for and handle missing values
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Ensure correct data types for datetime columns
        if self.datetime_format:
            for column in ['open_time', 'close_time']:
                if column in self.df.columns:
                    self.df[column] = pd.to_datetime(self.df[column], format=self.datetime_format, utc=True)
        else:
            for column in ['open_time', 'close_time']:
                if column in self.df.columns:
                    self.df[column] = pd.to_datetime(self.df[column], unit='ms', errors='coerce', utc=True)
                    
        # Batch conversion of data types
        for column, dtype in {
            'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float',
            'quote_asset_volume': 'float', 'number_of_trades': 'int', 'taker_buy_base_asset_volume': 'float',
            'taker_buy_quote_asset_volume': 'float', 'ignore': 'float'
        }.items():
            if column in self.df.columns:
                self.df[column] = self.df[column].astype(dtype)

        self.df.drop_duplicates(inplace=True)


    def resample_align(self):
        """
        Resample and align the DataFrame based on the specified period and truncate with custom labels.
        
        :param period: str, period for resampling ('D' for days, 'M' for months, etc.)
        :return: pd.DataFrame, the resampled, aligned, and truncated DataFrame
        """
        # Ensure the 'open_time' column is present and convert it to datetime
        period = self.params.get('resample_freq', 'D')
        if 'open_time' not in self.df.columns:
            raise KeyError("The DataFrame does not contain a 'open_time' column.")
        
        if self.datetime_format:
            self.df['open_time'] = pd.to_datetime(self.df['open_time'], format=self.datetime_format, utc=True)
        else:
            self.df['open_time'] = pd.to_datetime(self.df['open_time'], unit='ms', errors='coerce', utc=True)
        
        self.df.set_index('open_time', inplace=True)
        agg_funcs = {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
            'close_time': 'last', 'quote_asset_volume': 'sum', 'number_of_trades': 'sum',
            'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum', 'ignore': 'sum'
        }

        # Filter the aggregation functions to include only those columns that exist in the DataFrame
        agg_funcs = {col: func for col, func in agg_funcs.items() if col in self.df.columns}
        resampled_df = self.df.resample(period).agg(agg_funcs).ffill().bfill().drop_duplicates() # some data might be missing after resampling
        resampled_df.reset_index(inplace=True) # reset the index to make 'open_time' a column
        
        # Truncate the DataFrame to contain only the specified labels if provided
        truncated_df = resampled_df.reindex(columns=self.required_labels, fill_value=0.0).copy()
        # Handle missing columns by filling with default values (if necessary, might happen after resampling)
        for label in self.required_labels:
            if label not in truncated_df.columns:
                truncated_df[label] = pd.NaT if label in ['open_time', 'close_time'] else 0 if label == 'number_of_trades' else 0.0

        self.df = truncated_df

    def remove_outliers(self):
        """
        Remove outliers from the DataFrame using the Z-score method.
        """
        threshold = self.params.get('threshold', 20)
        numeric_df = self.df.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        filtered_df = self.df[(z_scores < threshold).all(axis=1)]
        self.df = filtered_df

    def change_time_zone(self):
        """
        Change the time zone of a datetime column in the DataFrame.
        
        Parameters:
        column_name (str): The name of the datetime column.
        utc_offset (int, optional): The UTC offset (e.g., +3 for UTC+3). Defaults to None (current local time zone).
        This function is not usually used in most cases, but it can be helpful for certain applications.
        """
        utc_offset = self.params.get('utc_offset', None)

        # Convert the column to datetime if it is not already
        for column_name in self.df.columns:
            if 'time' in column_name:
                self.df[column_name] = pd.to_datetime(self.df[column_name], utc=True)
        # Determine the target time zone
        if utc_offset is None:
            target_tz = dt.now().astimezone().tzinfo
        else:
            target_tz = pytz.FixedOffset(utc_offset * 60)

        for column_name in self.df.columns:
            if 'time' in column_name:
                self.df[column_name] = self.df[column_name].dt.tz_convert(target_tz)
                    
    def substitute_outliers(self):
        """
        Substitute outliers in the DataFrame with the mean value of the adjacent data points using the Z-score method.
        
        :param threshold: float, the Z-score threshold to identify outliers
        :param adjacent_count: int, the number of adjacent data points to consider for substitution
        """
        threshold = self.params.get('threshold', 20)
        adjacent_count = self.params.get('adjacent_count', 5)
        numeric_df = self.df.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outliers = z_scores > threshold
        half_adjacent = adjacent_count // 2

        def substitute_value(row, col):
            if row[col]:
                start_idx = max(0, row.name - half_adjacent)
                end_idx = min(len(self.df), row.name + half_adjacent + 1)
                adjacent_values = numeric_df[col].iloc[start_idx:end_idx].drop(row.name)
                return adjacent_values.mean()
            return self.df.at[row.name, col]

        for col in numeric_df.columns:
            self.df[col] = outliers[col].apply(lambda row: substitute_value(row, col), axis=1)
    
    def get_cleaned_df(self):
        """
        Return the cleaned DataFrame.
        :return: pd.DataFrame, the cleaned DataFrame
        """
        if self.params.get('check_labels', False):
            missing_labels = [label for label in self.required_labels if label not in self.df.columns]
            if missing_labels:
                raise ValueError(f"DataFrame does not have the required labels: {missing_labels}")

        # Clean the DataFrame if required
        if self.params.get('dtype', False):
            self.datatype_df()

        # Remove zero variance columns if required
        if self.params.get('zero_variance', False):
            zero_variance_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
            self.df.drop(columns=zero_variance_columns, inplace=True)
            
        # Remove outliers if required
        if self.params.get('remove_outliers', False):
            self.remove_outliers()

         # Substitute outliers if required
        if self.params.get('substitute_outliers', False):
            self.substitute_outliers()

        # Resample and align the DataFrame if required
        if self.params.get('resample_align', False):
            self.resample_align()
        
        # Change time zone if required
        if self.params.get('timezone', False):
            self.change_time_zone()

        return self.df


        
class DataChecker:
    def __init__(self, df, params, expected_types=None):
        """
        Base class for data checking.
        :param df: DataFrame containing cleaned data.
        :param params: Dictionary containing various parameters for data processing.
        :param expected_types: Dictionary containing expected data types for each column.
        """

        self.df = df
        self.params = params
        self.expected_types = expected_types

    def missing_check(self):
        """
        Check for missing values in the DataFrame and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :return: pd.DataFrame, the summary of missing values
        """
        missing_values = self.df.isnull().sum()
        missing_data = pd.DataFrame({'Missing Values': missing_values})
        is_clean = missing_values.sum() == 0

        return missing_data, is_clean

    def duplicate_check(self):
        """
        Check for duplicate rows in the DataFrame and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :return: pd.DataFrame, the summary of duplicate rows
        """
        duplicate_rows = self.df.duplicated().sum()
        duplicate_data = pd.DataFrame({'Duplicate Rows': [duplicate_rows]})
        is_clean = duplicate_rows == 0

        return duplicate_data, is_clean

    def outlier_check(self):
        """
        Check for outliers in the DataFrame using the Z-score method and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :param threshold: float, the Z-score threshold to identify outliers
        :return: pd.DataFrame, the summary of outliers
        """
        # Select only numeric columns
        threshold = self.params.get('threshold', 20)
        numeric_df = self.df.select_dtypes(include=[np.number])

        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outliers = (z_scores >= threshold).sum().sum()
        outlier_data = pd.DataFrame({'Outliers': [outliers]})
        is_clean = outliers == 0
        
        return outlier_data, is_clean

    def datatype_check(self):
        """
        Check the data types of the columns in the DataFrame and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :param expected_types: dict, the dictionary of expected data types (default is None)
        :return: dict, the summary of data types and a boolean indicating if the data types are as expected
        """
        # Default expected types
        if self.expected_types is None:
            self.expected_types = {
                'open_time': 'datetime64[ns, UTC]',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64',
                'close_time': 'datetime64[ns, UTC]',
                'quote_asset_volume': 'float64',
                'number_of_trades': 'int',
                'taker_buy_base_asset_volume': 'float64',
                'taker_buy_quote_asset_volume': 'float64',
            }
        
        data_types = self.df.dtypes
        data_type_data = pd.DataFrame({'Data Types': data_types})
        # Check only the columns that are present in both the DataFrame and the expected types
        common_columns = set(data_types.index).intersection(self.expected_types.keys())
        is_clean = all(data_types[col] == self.expected_types[col] for col in common_columns)
        
        return data_type_data, is_clean

    def logical_check(self):
        """
        Check for logical errors in the DataFrame and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :return: dict, the summary of logical errors and a boolean indicating if the data is clean
        """
        logical_errors = {
            'Negative Open Prices': (self.df['open'] < 0).sum(),
            'Negative Close Prices': (self.df['close'] < 0).sum(),
            'Negative High Prices': (self.df['high'] < 0).sum(),
            'Negative Low Prices': (self.df['low'] < 0).sum()
        }
        logical_data = pd.DataFrame({'Logical Errors': [logical_errors]})
        
        is_clean = True if all(value == 0 for value in logical_errors.values()) else False
        
        return logical_data, is_clean

    def zero_variance_check(self):
        """
        Check for columns with zero variance in the DataFrame and return a summary.
        
        :param df: pd.DataFrame, the DataFrame to check
        :return: pd.DataFrame, the summary of columns with zero variance
        """
        zero_variance_columns = self.df.columns[self.df.nunique() == 1]
        zero_variance_data = pd.DataFrame({'Zero Variance Columns': [zero_variance_columns]})
        is_clean = True if len(zero_variance_columns) == 0 else False

        return zero_variance_data, is_clean
    
    def perform_check(self):
        """
        Perform all checks and return a summary of the results.
        :return: dict, the summary of all checks and a boolean indicating if the data is clean
        """
        results = {}
        results['size'] = self.df.shape
        is_clean = True
        
        if self.params.get('missing_check', False):
            missing_result, clean = self.missing_check()
            results['missing_check'] = missing_result
            is_clean = is_clean and clean
            if not clean:
                print("Missing values found in the DataFrame.")

        if self.params.get('duplicate_check', False):
            duplicate_result, clean = self.duplicate_check()
            results['duplicate_check'] = duplicate_result
            is_clean = is_clean and clean
            if not clean:
                print("Duplicate rows found in the DataFrame.")
        
        if self.params.get('outlier_check', False):
            outlier_result, clean = self.outlier_check()
            results['outlier_check'] = outlier_result
            is_clean = is_clean and clean
            if not clean:
                print("Outliers found in the DataFrame.")

        if self.params.get('datatype_check', False):
            datatype_result, clean = self.datatype_check()
            results['datatype_check'] = datatype_result
            is_clean = is_clean and clean
            if not clean:
                print("Data types are not as expected in the DataFrame.")
                print(self.df.dtypes)
        
        if self.params.get('logical_check', False):
            logical_result, clean = self.logical_check()
            results['logical_check'] = logical_result
            is_clean = is_clean and clean
            if not clean:
                print("Logical errors found in the DataFrame.")
        
        if self.params.get('zero_variance_check', False):
            zero_variance_result, clean = self.zero_variance_check()
            results['zero_variance_check'] = zero_variance_result
            is_clean = is_clean and clean
            if not clean:
                print("Columns with zero variance found in the DataFrame.")
        
        results['is_clean'] = is_clean
        
        return results
    
    def print_check(self, results):

        """
        Print the results of the sanity checks in an intuitive way.
        
        :param results: dict, the results of the sanity checks
        """
        for check, result in results.items():
            print(f"\n{check.replace('_', ' ').title()}:\n")
            if isinstance(result, pd.DataFrame):
                print(result.to_string(index=False))
            else:
                print(result)

