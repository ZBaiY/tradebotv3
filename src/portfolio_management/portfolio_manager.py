# portfolio_manager.py

class PortfolioManager:
    def __init__(self, allocated_capital, target_allocation):
        """
        Parameters:
            allocated_capital (float): The total capital allocated to crypto investments.
            target_allocation (dict): A dictionary mapping each crypto symbol to a target allocation percentage.
        """
        self.allocated_capital = allocated_capital
        self.target_allocation = target_allocation

    def calculate_individual_allocations(self):
        """
        Divides the allocated capital based on the target allocation for each crypto.
        """
        allocations = {}
        for symbol, percentage in self.target_allocation.items():
            allocations[symbol] = self.allocated_capital * percentage
        return allocations

    def rebalance(self, current_holdings):
        """
        Rebalance portfolio based on the target allocation and current holdings.
        
        Parameters:
            current_holdings (dict): Current holdings by crypto symbol.

        Returns:
            rebalance_orders (dict): Orders needed to achieve target allocation.
        """
        target_allocations = self.calculate_individual_allocations()
        rebalance_orders = {}
        for symbol, target_amount in target_allocations.items():
            current_amount = current_holdings.get(symbol, 0)
            rebalance_orders[symbol] = target_amount - current_amount
        return rebalance_orders

