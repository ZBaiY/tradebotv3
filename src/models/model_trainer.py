# base_model.py
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
## historical data handler for backtesting
from src.data_handling.historical_data_handler import HistoricalDataHandler
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor
import src.signal_processing.filters as filter
import src.signal_processing.transform as transform
import numpy as np
import pandas as pd

