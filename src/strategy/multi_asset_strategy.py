import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.strategy.base_strategy import BaseStrategy
from src.strategy.single_asset_strategy import SingleAssetStrategy

"""This Module will contains the prediction and models for multiple assets."""
"""
The risk manager as a instance in this class will only be used for 
to apply strategies that generate signals for the multiple assets.
Strategy class by it self will not do the risk management calculations.
"""


class MultiAssetStrategy(BaseStrategy):
    def __init__(self, equity, balances, allocation_cryp, assigned_percentage, datahandler, risk_manager, m_config = None, d_config = None, feature_module=None, signal_processors=None):
        """
        Strategy class for trading
        multiple assets simultaneously.
        there might be multiple signal processors
        signal_processors: [signal_processor1, signal_processor2, ...]
        m_config: model configuration
        s_config: strategy configuration
        """
        super().__init__(equity, balances, allocation_cryp, assigned_percentage, datahandler, risk_manager, feature_module, signal_processors)
        self.signals = {}
        self.m_config = m_config
        self.d_config = d_config
        self.strategies = {}
        
    def initialize_singles(self):
        for symbol in self.symbols:
            self.strategies[symbol] = SingleAssetStrategy(symbol, self.m_config[symbol], self.d_config[symbol], self.risk_manager.risk_managers[symbol], self.data_handler, self.signal_processors, self.features)
            self.strategies[symbol].initialize(self.risk_manager.risk_managers[symbol])

    def pre_run(self):
        rebanlance_need = self.check_rebalance()
        if rebanlance_need:
            self.rebalance_portfolio()

    def run_strategy_market(self):
        """
        Run prediction on the given data.
        """
        for symbol in self.symbols:
            # data = self.data_handler.cleaned_data
            self.strategies[symbol].run_prediction()
            self.signals[symbol] = self.strategies[symbol].make_decision_market()
        return self.signals
    
    def get_signals(self):
        return self.signals
    
    def update_equity_balance(self, equity, balances, trade = False):
        self.equity = equity
        self.balances = balances
        for symbol in self.symbols:
            self.strategies[symbol].set_equity(equity)
            self.strategies[symbol].set_balance(balances[symbol],trade)



########### I believe the equity, balances, assigned_percentage, and allocation_cryp are handled by the risk manager
########### Maybe can be used to dobule check the signals are within the assigned percentage
########### Maybe do the rebalance here
########### They are abit redundant here, check later

    def update_equity(self, equity):
        self.equity = equity
        for symbol in self.symbols:
            self.strategies[symbol].set_equity(equity)
    
    def update_balances(self, balances, trade = False):
        self.balances = balances
        for symbol in self.symbols:
            self.strategies[symbol].set_balances(balances)
    
    def update_assigned_percentage(self, assigned_percentage):
        self.assigned_percentage = assigned_percentage
        for symbol in self.symbols:
            self.strategies[symbol].set_assigned_percentage(assigned_percentage)

    
    def check_rebalance(self):
        """
        Check if rebalancing is needed.
        """
        return False


        
