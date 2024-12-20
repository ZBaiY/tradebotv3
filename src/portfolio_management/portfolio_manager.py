"""Should be directly called by the real-time trading module to allocate capital to crypto."""
"""Not a part of the strategy module."""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

# portfolio_manager.py

class PortfolioManager:
    def __init__(self, equity, balances, allocation_cryp, symbols, path='config/portfolio.json'):
        """
        Parameters:
            allocated_capital (float): The total capital allocated to crypto investments.
            target_allocation (dict): A dictionary mapping each crypto symbol to a target allocation percentage.
        """
        self.equity = equity
        self.balances = balances
        self.symbols = symbols  
        self.allocation_cryp = allocation_cryp
        self.assigned_percentage = {}
        self.path = path
        self.config = json.load(open(self.path, 'r'))
        self.method = self.config['method']
        self.params = self.config['params']
        self.assigned_percentage = {}
        self.pre_run()
    
    def pre_run(self):
        if self.method == 'equal_weight':
            self.equal_weight()
        else:
            raise ValueError("Invalid method specified in the portfolio configuration file.")
    


    def set_equity(self, equity):
        self.equity = equity

    def set_balances(self, balances):
        self.balances = balances


    def equal_weight(self):
        """
        Calculates the target allocation for each crypto symbol based on equal weight.
        """
        num_symbols = len(self.symbols)
        self.assigned_percentage = {symbol: 1 / num_symbols for symbol in self.symbols}
        


    def rebalance(self, current_holdings):
        """
        Rebalance portfolio based on the target allocation and current holdings.
        
        Parameters:
            current_holdings (dict): Current holdings by crypto symbol.

        Returns:
            rebalance_orders (dict): Orders needed to achieve target allocation.
        """
        target_allocations = self.equal_weight()
        rebalance_orders = {}
        for symbol, target_amount in target_allocations.items():
            current_amount = current_holdings.get(symbol, 0)
            rebalance_orders[symbol] = target_amount - current_amount
        return rebalance_orders

    def get_assigned_percentage(self):
        self.equal_weight()
        return self.assigned_percentage