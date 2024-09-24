"""
fetch 1 week data and update the scaler
Run this file every week
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_handling.data_handler import DataHandler, ScalerHandler, update_scaler_with_recent_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import joblib
import os
import re
import json
from datetime import datetime

def process_files_in_folder(folder_path, required_labels, scaler_type, scaler_save_base_path, update_frequency=672):
    """
    Scans the given folder for files matching the pattern and applies the ScalerHandler to each file.
    
    :param folder_path: The folder path containing the historical data files.
    :param required_labels: List of columns to apply scaling on.
    :param scaler_type: Type of scaler ('minmax' or 'standard').
    :param scaler_save_base_path: Base path to save the scalers, organized by symbol and interval.
    :param update_frequency: Number of data points to use for fitting the scaler (default is 672 for weekly updates).
    """
    # Pattern to match file names like {symbol}_{start_date}_{end_date}_{interval}.{file_type}
    file_pattern = re.compile(r'(?P<symbol>\w+)_(?P<start_date>\d{4}-\d{2}-\d{2})_(?P<end_date>\d{4}-\d{2}-\d{2})_(?P<interval>\d+\w+)\.(?P<file_type>\w+)')

    # Scan the folder for files matching the pattern
    for file_name in os.listdir(folder_path):
        match = file_pattern.match(file_name)
        if match:
            # Extract information from the file name
            symbol = match.group('symbol')
            start_date = match.group('start_date')
            end_date = match.group('end_date')
            interval = match.group('interval')
            file_type = match.group('file_type')

            # Log information about the file being processed
            print(f"Processing file: {file_name}")
            print(f"Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}, Interval: {interval}, File Type: {file_type}")

            # Define the full path to the file
            file_path = os.path.join(folder_path, file_name)

            # Create a folder to save the scaler models organized by symbol and interval
            scaler_save_path = os.path.join(scaler_save_base_path, symbol, interval)
            os.makedirs(scaler_save_path, exist_ok=True)

            # Create a ScalerHandler instance for this symbol and interval
            scaler_handler = ScalerHandler(
                symbol=symbol,
                required_labels=required_labels,
                scaler=scaler_type,
                history_path=folder_path,
                scaler_save_path=scaler_save_path,
                update_frequency=update_frequency
            )
            path = os.path.join(folder_path, file_name)
            # Fit and save the scaler using the historical data from this file
            scaler_handler.fit_and_save_scaler(path=path)
        else:
            print(f"File '{file_name}' does not match the expected pattern and will be skipped.")

def first_update():
    folder_path = 'data/historical/processed/for_train'  # Input folder path
    scaler_save_base_path = 'data/scaler'  # Base path to save the scaler models
    json_file = 'config/fetch_real_time.json'  # JSON config file

    #print the json_file's full path

    # Read the required labels from the config file)
    with open(json_file, 'r') as file:
        fetch_config = json.load(file)
    
    required_labels = fetch_config['required_labels']
    scaler_type = fetch_config['scaler']
    
    # Process files in the folder and organize the saved scalers by symbol and interval
    process_files_in_folder(
        folder_path=folder_path,
        required_labels=required_labels,
        scaler_type=scaler_type,
        scaler_save_base_path=scaler_save_base_path,
        update_frequency=672  # Weekly updates
    )



def update():

    scaler_save_base_path = 'data/scaler'  # Base path to save the scaler models
    json_file = 'config/fetch_real_time.json'  # JSON config file

    #print the json_file's full path

    # Read the required labels from the config file)
    with open(json_file, 'r') as file:
        fetch_config = json.load(file)
    symbols = fetch_config['symbols']
    required_labels = fetch_config['required_labels']
    scaler_type = fetch_config['scaler']
    interval = fetch_config['interval']

    # Required labels (columns in the data)
    required_labels = ['open', 'high', 'low', 'close', 'volume']

    # Update scalers for the symbols
    update_scaler_with_recent_data(
        symbols=symbols,
        required_labels=required_labels,
        interval=interval,  # Specify the interval (e.g., '15m', '1h')
        scaler_type=scaler_type,
        scaler_save_base_path= scaler_save_base_path,  # Path where scalers are saved
        recent_data_path='data/recent',  # Path where recent data CSVs are stored
        update_frequency=500  # Fetch the most recent 500 data points (e.g., weekly update)
    )


if __name__ == '__main__':
    update()