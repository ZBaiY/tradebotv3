import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABC, abstractmethod
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler

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
        self.data_handler = datahandler
        self.risk_manager = risk_manager        
        self.symbols = None
        self.signal_processor = signal_processor
        self.features = feature_extractor
        self.allocation = None
        self.cryp_dist = None

        
        
    

    def set_symbols(self, symbols):
        self.symbols = symbols

    def set_allocation_cryp(self, allocation_cryp):
        self.allocation = allocation_cryp
    
    def set_crypto_dist(self, cryp_dist):
        ### cryp_dist is a dictionary with the crypto symbol as the key and the percentage of the portfolio as the value
        self.cryp_dist = cryp_dist

    def set_equity(self, equity):
        self.equity = equity

    def set_balances(self, balances):
        self.balances = balances

    def initialize(self):
        """Initialize strategy parameters and any necessary setup."""
        pass

    def update(self, market_data):
        """
        Update strategy with new market data.
        :param market_data: Dataframe or dictionary of recent market data
        """
        pass

    def calculate_signals(self):
        """
        Calculate buy/sell signals based on predictions and processed data.
        Returns the signals in a suitable format (e.g., dictionary or custom object).
        """
        pass

    def execute_trade(self, trade_signal):
        """
        Executes a trade based on the trade signal.
        :param trade_signal: Signal indicating action ('buy'/'sell') and quantity
        """
        # Example: Interact with live_trading module
        # live_trading.execute_order(trade_signal)
        print(f"Executing trade: {trade_signal}")
        
    def apply_str(self, trade_signal):
        """
        Adjust trade signal according to stop loss and take profit from portfolio manager.
        :param trade_signal: Signal to be adjusted
        """
        stop_loss, take_profit = self.risk_manager.calculate_str()
        # Apply SL/TP logic to the trade_signal here

    def check_risk(self):
        """
        Check if the current risk level is within the acceptable range.
        """
        pass

    def check_rebalance(self):
        """
        Check if the portfolio needs rebalancing based on the current asset allocation.
        """
        pass

    def rebalance_portfolio(self):
        """
        This function will be called periodically to rebalance the portfolio based on the current asset allocation.
        Rebalance the portfolio based on the current asset allocation.
        """
        pass