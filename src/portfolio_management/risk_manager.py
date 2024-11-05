# risk_manager.py

class RiskManager:
    def __init__(self, stop_loss_threshold=0.1, take_profit_threshold=0.2):
        """
        Parameters:
            stop_loss_threshold (float): Loss threshold to trigger stop-loss.
            take_profit_threshold (float): Profit threshold to trigger take-profit.
        """
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold 

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
