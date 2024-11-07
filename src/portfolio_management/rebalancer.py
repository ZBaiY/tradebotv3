"""
This module just executes rebalancing of the portfolio based on the threshold.
Which is guided by the PortfolioManager class.
"""

# rebalancer.py
from portfolio_manager import PortfolioManager

class Rebalancer:
    def __init__(self, portfolio_manager, threshold=0.05):
        """
        Parameters:
            portfolio_manager (PortfolioManager): Instance of PortfolioManager for rebalancing.
            threshold (float): Allowed deviation from the target allocation before rebalancing.
        """
        self.portfolio_manager = portfolio_manager
        self.threshold = threshold

    def needs_rebalance(self, current_holdings):
        """
        Determines if the portfolio needs rebalancing based on threshold.
        
        Parameters:
            current_holdings (dict): Current holdings by crypto symbol.

        Returns:
            bool: True if rebalancing is needed, otherwise False.
        """
        rebalance_orders = self.portfolio_manager.rebalance(current_holdings)
        for symbol, adjustment in rebalance_orders.items():
            if abs(adjustment / self.portfolio_manager.allocated_capital) > self.threshold:
                return True
        return False

    def execute_rebalance(self, current_holdings):
        """
        Executes rebalancing by calculating orders and returning the required trades.

        Parameters:
            current_holdings (dict): Current holdings by crypto symbol.

        Returns:
            dict: Rebalance orders needed to achieve the target allocation.
        """
        if self.needs_rebalance(current_holdings):
            return self.portfolio_manager.rebalance(current_holdings)
        return {}
