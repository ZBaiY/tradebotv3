import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_handling.historical_data_handler import HistoricalDataHandler  # Assuming this is the module where the class resides

if __name__ == '__main__':
    
    # File paths to parameter files
    source_file = os.path.join('config', 'source.json')                  # Example: source config file
    cleaner_file = os.path.join('config', 'cleaner.json')  # Example: cleaner config file
    checker_file = os.path.join('config', 'checker.json')  # Example: checker config file
    fetch_data_file = os.path.join('config', 'fetchtest.json')         # JSON fetch configuration

    # Initialize the HistoricalDataHandler
    historical_data_handler = HistoricalDataHandler(
        source_file=source_file,
        cleaner_file=cleaner_file,
        checker_file=checker_file
    )

    # Fetch and save data based on the JSON configuration
    historical_data_handler.fetch_save_json(fetch_data_file)
