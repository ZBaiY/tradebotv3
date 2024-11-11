import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABC, abstractmethod
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler

"""This is the base class for all strategies, mostly handling the allocations and risk management."""


class BaseStrategy():
    def __init__(self, datahandler, risk_manager, feature_extractor=None, signal_processor=None):
        """
        Base class for all strategies.

        :param model: Prediction model module instance
        :param portfolio_manager: Portfolio management module instance
        :param feature_module: Optional feature module for data processing
        :param signal_processor: Optional signal processing module for processed data
        """
        self.equity = None
        self.balances = None
        self.assigned_calpitals = {}
        self.data_handler = datahandler
        self.risk_manager = risk_manager        
        self.symbols = None
        self.signal_processor = signal_processor
        self.features = feature_extractor
        self.allocation = None
        self.cryp_dist = None
        self.symbols = self.data_handler.symbols


    def set_allocation_cryp(self, allocation_cryp):
        self.allocation = allocation_cryp
    
    def set_crypto_dist(self, cryp_dist):
        ### cryp_dist is a dictionary with the crypto symbol as the key and the percentage of the portfolio as the value
        self.cryp_dist = cryp_dist

    
    def set_equity(self, equity):
        self.equity = equity

    def set_balances(self, balances):
        self.balances = balances

    def set_assigned_capitals(self, assigned_capitals):
        self.assigned_calpitals = assigned_capitals

    def initialize(self):
        """Initialize strategy parameters and any necessary setup."""
        pass

    def update(self, market_data):
        """
        Update strategy with new market data.
        :param market_data: Dataframe or dictionary of recent market data
        """
        pass




