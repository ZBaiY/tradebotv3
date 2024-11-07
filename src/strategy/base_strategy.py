import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABC, abstractmethod
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler

class BaseStrategy(ABC):
    def __init__(self, datahandler, portfolio_manager, feature_module=None, signal_processor=None):
        """
        Base class for all strategies.

        :param model: Prediction model module instance
        :param portfolio_manager: Portfolio management module instance
        :param feature_module: Optional feature module for data processing
        :param signal_processor: Optional signal processing module for processed data
        """
        self.equity = None
        self.balances = None
        self.portfolio_manager = portfolio_manager
        self.feature_module = feature_module
        self.data_handler = datahandler
        self.symbols = self.data_handler.symbols
        self.signal_processor = signal_processor

    def set_equity(self, equity):
        self.equity = equity
    def set_balances(self, balances):
        self.balances = balances

    @abstractmethod
    def initialize(self):
        """Initialize strategy parameters and any necessary setup."""
        pass

    @abstractmethod
    def update(self, market_data):
        """
        Update strategy with new market data.
        :param market_data: Dataframe or dictionary of recent market data
        """
        pass

    @abstractmethod
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
        
    def apply_stop_loss_take_profit(self, trade_signal):
        """
        Adjust trade signal according to stop loss and take profit from portfolio manager.
        :param trade_signal: Signal to be adjusted
        """
        stop_loss, take_profit = self.portfolio_manager.get_stop_loss_take_profit()
        # Apply SL/TP logic to the trade_signal here
