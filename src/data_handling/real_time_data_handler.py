"""
The class of realtime datahandler
"""

from data_handler import DataHandler

class RealTimeDataHandler(DataHandler):
    def __init__(self, source, frequency):
        super().__init__(source, frequency)

    def fetch_data(self, symbol):
        # Code to retrieve real-time data for the given symbol
        pass

    def clean_real_time_data(self, data):
        # Code to clean real-time data (potentially different from historical)
        pass
