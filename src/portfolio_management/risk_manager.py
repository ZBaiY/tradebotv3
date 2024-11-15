# risk_manager.py
import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler
import src.portfolio_management.single_risk as srMan

class RiskManager:
    def __init__(self, config, data_handler, signal_processor=None, feature_handler=None):
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
        self.assigned_calpitals = {}
        self.data_handler = data_handler
        self.symbols = self.data_handler.symbols
        self.signal_processor = signal_processor
        self.feature_handler = feature_handler
        self.stop_loss_threshold = None
        self.take_profit_threshold = None
        self.entry_price = {}
        self.config = config
        self.risk_managers = {}
        self.initialize_risk_managers()

    def set_equity(self, equity):
        self.equity = equity

    def set_balances(self, balances):
        self.balances = balances

    def set_assigned_capitals(self, assigned_capitals):
        self.assigned_calpitals = assigned_capitals

    def update_balances(self, balances):
        self.balances = balances
        for symbol in self.symbols:
            self.risk_managers[symbol].set_balance(self.balances[symbol])
    def update_equity(self, equity):
        self.equity = equity

    def update_assigned_capitals(self, assigned_capitals):
        self.assigned_calpitals = assigned_capitals
        for symbol in self.symbols:
            self.risk_managers[symbol].set_assigned_capital(self.assigned_calpitals[symbol])

    
    def initialize_singles(self):
        for symbol in self.symbols:
                self.risk_managers[symbol] = srMan.SingleRiskManager(symbol, self.config[symbol]["risk_manager"], self.data_handler, self.signal_processor, self.feature_handler)
                self.risk_managers[symbol].set_balance(self.balances[symbol])
                self.risk_managers[symbol].set_assigned_capital(self.assigned_calpitals[symbol])
    
        
    
    def set_entry_price(self, entry_price):
        self.entry_price = entry_price
        for symbol in self.symbols:
            self.risk_managers[symbol].set_entry_price(entry_price[symbol])
    
    ### Calculate stop-loss take-profit and postion size
    def calculate_stp(self, symbol): 
        for symbol in self.symbols:
            self.risk_managers[symbol].calculate_stop_loss()
            self.risk_managers[symbol].calculate_take_profit()
            self.risk_managers[symbol].calculate_position_size(self.equity, self.balances)

    def request_prediction(self, predictions):
        for symbol in self.symbols:
            self.risk_managers[symbol].request_prediction(predictions[symbol])


    def update_risk_parameters(self, new_stop_loss, new_take_profit):
        """
        Updates stop-loss and take-profit parameters.
        """
        self.stop_loss_threshold = new_stop_loss
        self.take_profit_threshold = new_take_profit
        

    def get_stop_loss(self):
        stop_losses = {}
        for symbol in self.symbols:
            stop_losses[symbol] = self.risk_managers[symbol].get_stop_loss()
        return stop_losses

    def get_take_profit(self):
        take_profits = {}
        for symbol in self.symbols:
            take_profits[symbol] = self.risk_managers[symbol].get_take_profit()
        return take_profits
    
    def get_position_size(self):
        position_sizes = {}
        for symbol in self.symbols:
            position_sizes[symbol] = self.risk_managers[symbol].get_position_size()
        return position_sizes


