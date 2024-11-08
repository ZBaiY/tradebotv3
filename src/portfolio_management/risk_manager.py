# risk_manager.py
import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler
import src.portfolio_management.single_risk as srMan

class RiskManager:
    def __init__(self, data_handler, config):
        """
        Parameters:
            stop_loss_threshold (float): Loss threshold to trigger stop-loss.
            take_profit_threshold (float): Profit threshold to trigger take-profit.
            config (dict): Configuration parameters.
            {
            "symbol1": {"method": "simple", "stop_loss": 0.05, "take_profit": 0.1},
            "symbol2": {"method": "atr", "stop_loss": 0.05, "take_profit": 0.1, "atr_window": 14}
            }
        """
        self.equity = None
        self.balances = None
        self.data_handler = data_handler
        self.symbols = self.data_handler.symbols
        self.stop_loss_threshold = None
        self.take_profit_threshold = None

    def set_equity(self, equity):
        self.equity = equity
    def set_balances(self, balances):
        self.balances = balances

    
    def evaluate_position(self, symbol, entry_price, current_price):
        """
        Evaluates whether to close a position based on stop-loss or take-profit levels.
        
        Parameters:
            symbol (str): Crypto symbol.
            entry_price (float): The price at which the position was opened.
            current_price (float): The current market price.

        Returns:
            str: 'sell' if conditions for stop-loss or take-profit are met, 'hold' otherwise.
        """
        if current_price <= entry_price * (1 - self.stop_loss_threshold):
            return 'sell'  # Stop-loss
        elif current_price >= entry_price * (1 + self.take_profit_threshold):
            return 'sell'  # Take-profit
        else:
            return 'hold'
    
    def update_risk_parameters(self, new_stop_loss, new_take_profit):
        """
        Updates stop-loss and take-profit parameters.
        """
        self.stop_loss_threshold = new_stop_loss
        self.take_profit_threshold = new_take_profit
        

    def get_stop_loss(self):
        pass
    def get_take_profit(self):
        pass


