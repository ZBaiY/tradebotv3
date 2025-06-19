import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_handling.historical_data_handler import HistoricalDataHandler  # Assuming this is the module where the class resides


# -----------------------------------------------------------------------------  
# Public entry point for programmatic use (e.g. pytest, notebooks, other scripts)
# -----------------------------------------------------------------------------
def main(fetch_cfg: str = os.path.join('config', 'fetch_data.json')):

    # File paths to parameter files
    source_file   = os.path.join('config', 'source.json')
    cleaner_file  = os.path.join('config', 'cleaner.json')
    checker_file  = os.path.join('config', 'checker.json')

    # Initialize the HistoricalDataHandler
    hdh = HistoricalDataHandler(
        source_file=source_file,
        cleaner_file=cleaner_file,
        checker_file=checker_file,
    )

    # Fetch and save data based on the JSON configuration
    hdh.fetch_save_json(fetch_cfg)


if __name__ == "__main__":
    main()
