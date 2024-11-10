"""
This is the class for single asset trading risk management.
"""
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class SingleRiskManager:
    def __init__(self, symbol, config):
        """
        Parameters:
            symbol (str): Crypto symbol.
            config (dict): Configuration parameters.
            {
            "method": "simple",
            "stop_loss": 0.05,
            "take_profit": 0.1
            }
        """
        self.symbol = symbol
        self.stop_method = config.get("stop_method", "static")
        self.stop_params = config.get("stop_params", {})
        self.take_method = config.get("take_method", "static")
        self.take_params = config.get("take_params", {})
        self.position_method = config.get("position_method", "static")
        self.postion_params = config.get("position_params", {})
        self.entry_price = None
        
        self.setup_stop()
        self.setup_take()
        self.setup_position()
        self.required_stop_fields = None
        self.required_take_fields = None
        self.required_position_fields = None
        self.input_stop = None  
        self.input_take = None
        self.input_position = None
        self.take_profit = None
        self.stop_loss = None

    def read_json_file(file_path):
        """
        Read a JSON file and return its contents as a Python object.
        :param file_path: The path to the JSON file.
        :return: The contents of the JSON file as a Python object.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return None
        
    def set_entry_price(self, entry_price):
        self.entry_price = entry_price

    def setup_stop(self):
        stop_requires = self.read_json_file("src/portfolio_management/model_details/stop_required_fields.json")

        if self.stop_method == "atr":
            self.atr_multiplier = self.stop_params.get("atr_multiplier", 1)
            self.required_stop_fields = stop_requires["atr"]
        elif self.stop_method == "trailing":
            self.trail_percent = self.stop_params.get("trail_percent", 0.05)
            self.trail_lookback = self.stop_params.get("trail_lookback", 15)
            self.required_stop_fields = stop_requires["trailing"]
        elif self.stop_method == "static":
            self.stop_loss_percent = self.stop_params.get("stop_loss_percent", 0.02)
            self.required_stop_fields = stop_requires["static"]
        elif self.stop_method == "risk_reward":
            self.risk_reward_ratio = self.stop_params.get("risk_reward_ratio", 2)
            self.required_stop_fields = stop_requires["risk_reward"]
        elif self.stop_method == "custom_1":
            self.atr_multiplier= self.stop_params.get("atr_multiplier", 1)
            self.trail_percent = self.stop_params.get("trail_percent", 0.2)
            self.trail_lookback = self.stop_params.get("trail_lookback", 15)
            self.stop_loss_percent=self.stop_params.get("stop_loss_percent", 0.02)
            self.required_stop_fields = stop_requires["custom_1"]
        else:
            raise ValueError("Invalid method for stop loss.")
        
        
    def setup_take(self): 
        take_requires = self.read_json_file("src/portfolio_management/model_details/take_required_fields.json")

        if self.take_method == "static":
            self.take_profit_percent = self.take_params.get("take_profit_percent", 0.05)
            self.required_take_fields = take_requires["static"]
        elif self.take_method == "trailing":
            self.trail_percent = self.take_params.get("trail_percent", 0.1)
            self.required_take_fields = take_requires["trailing"]
        elif self.take_method == "risk_reward":
            self.risk_reward_ratio = self.take_params.get("risk_reward_ratio", 2)
            self.required_take_fields = take_requires["risk_reward"]
        else:
            raise ValueError("Invalid method for take profit.")
    
    def setup_position(self):
        if self.position_method == "kelly":
            self.win_rate = self.postion_params.get("win_rate", 0.6)
            self.avg_win = self.postion_params.get("avg_win", 100)
            self.avg_loss = self.postion_params.get("avg_loss", 50)
        elif self.position_method == "risk_based":
            self.risk_per_trade = self.postion_params.get("risk_per_trade", 0.01)
        elif self.position_method == "fixed_fractional":
            self.risk_percentage = self.postion_params.get("risk_percentage", 2)
        else:
            raise ValueError("Invalid method for position sizing.")
        

    def request_data(self, datahandler, features, signal_processor):
        # The requested data will be written in files under the folder src/model_details/...
        if "current_price" in self.required_stop_fields:
            self.input_stop['current_price'] = datahandler.get_last_price(self.symbol)['close']
        if "atr" in self.required_stop_fields:
            self.input_stop['atr'] = features.get_last_indicator(self.symbol, 'atr')
        if "max_price" in self.required_stop_fields:
            look_back_prices = datahandler.get_data_limit(self.symbol, self.trail_lookback)['close']
            self.input_stop['max_price'] = max(look_back_prices)
        if "entry_price" in self.required_stop_fields:
            self.input_stop['entry_price'] = self.entry_price
        
        if "current_price" in self.required_take_fields:
            self.input_take['current_price'] = datahandler.get_last_price(self.symbol)['close']
        if "max_price" in self.required_take_fields:
            look_back_prices = datahandler.get_data_limit(self.symbol, self.trail_lookback)['close']
            self.input_take['max_price'] = max(look_back_prices)
        if "entry_price" in self.required_take_fields:
            self.input_take['entry_price'] = self.entry_price
        if "current_price" in self.required_position_fields:
            self.input_position['current_price'] = datahandler.get_last_price(self.symbol)['close']
        
    def calculate_stop_loss(self):
        if self.stop_method == "atr":
            self.stop_loss = StopLoss.atr_stop_loss(self.atr_multiplier, **self.input_stop)
        elif self.stop_method == "trailing":
            self.stop_loss = StopLoss.trailing_stop_loss(self.trail_percent, **self.input_stop)
        elif self.stop_method == "static":
            self.stop_loss = StopLoss.static_stop_loss(self.stop_loss_percent, **self.input_stop)
        elif self.stop_method == "risk_reward":
            self.stop_loss = StopLoss.risk_reward_take_profit(self.risk_reward_ratio, **self.input_stop)
        elif self.stop_method == "custom_1":
            ###### This needs to be updated (trend_dir) ######
            self.stop_loss = StopLoss.custom_1(self.atr_multiplier, self.trail_percent, self.stop_loss_percent, **self.input_stop)
        else:
            raise ValueError("Invalid method for stop loss.")
    
    def calculate_take_profit(self):
        self.input_take['stop_loss_price'] = self.stop_loss
        if self.take_method == "static":
            self.take_profit = TakeProfit.static_take_profit(self.take_profit_percent, **self.input_take)
        elif self.take_method == "trailing":
            self.take_profit = TakeProfit.trailing_take_profit(self.trail_percent, **self.input_take)
        elif self.take_method == "risk_reward":
            self.take_profit = TakeProfit.risk_reward_take_profit(self.risk_reward_ratio, **self.input_take)
        else:
            raise ValueError("Invalid method for take profit.")
        

    ######### Position Sizing needs updates #########
    def calculate_position_size(self):
        self.input_position['stop_loss_price'] = self.stop_loss
        self.input_position['take_profit_price'] = self.take_profit
        if self.position_method == "kelly":
            self.position_size = PositionSizing.kelly_criterion(self.win_rate, self.avg_win, self.avg_loss)
        elif self.position_method == "risk_based":
            self.position_size = PositionSizing.risk_based_position_size(self.risk_per_trade, **self.input_position)
        elif self.position_method == "fixed_fractional":
            self.position_size = PositionSizing.fixed_fractional_position_size(self.risk_percentage, **self.input_position)
        else:
            raise ValueError("Invalid method for position sizing.")

    def get_stop_loss(self):
        return self.stop_loss
    def get_take_profit(self):
        return self.take_profit
    def get_position_size(self):
        return self.position_size

    
        
        


        


        






class StopLoss:

    @staticmethod
    def atr_stop_loss(atr_multiplier=1, **kwargs):
        """
        Calculate the stop loss based on the average true range (ATR).

        :param current_price: The current market price.
        :param atr: The average true range (ATR) value.
        :param atr_multiplier: The ATR multiplier for the stop loss.
        :return: The stop loss price.
        """
        current_price = kwargs.get("current_price", None)
        atr = kwargs.get("atr", None)
        if current_price is None or atr is None:
            return -1
        stop_loss = max(current_price - (atr_multiplier * atr), 0)
        return stop_loss
    
    @staticmethod
    def trailing_stop_loss(trail_percent=0.15, **kwargs):
        """
        Calculates the trailing stop-loss price.
        
        Parameters:
        - current_price (float): The current market price.
        - max_price (float): The highest price since entry.
        - trail_percentage (float): The percentage distance for the trailing stop (e.g., 0.02 for 2%).
        
        Returns:
        - float: The trailing stop-loss price.
        """
        max_price = kwargs.get("max_price", None)
        if max_price is None:
            return -1
        trail_amount = max_price * trail_percent
        stop_loss = max_price - trail_amount
        return max(stop_loss,0)  # Ensure stop loss doesn't go below entry price

    @staticmethod
    def static_stop_loss(stop_loss_percent=0.02, **kwargs):
        """
        Calculate static stop-loss price based on a percentage.

        :param entry_price: Entry price of the trade.
        :param stop_loss_percent: Percentage below entry price for stop loss.
        :return: Stop-loss price.
        """
        entry_price = kwargs.get("entry_price", None)
        if entry_price is None:
            return -1
        stop_loss_price = entry_price * (1 - stop_loss_percent)
        return max(stop_loss_price,0)

    @staticmethod
    def custom_1(trend_dir, atr_multiplier=1, trail_percent = 0.2, stop_loss_percent=0.02, **kwargs):
        """
        Calculate the stop loss based on the trend direction.
        atr_stop_loss for uptrend, trailing_stop_loss for downtrend, static_stop_loss for sideway trend.

        :param trend_dir: The direction of the trend (-1 for downtrend, 0 for sideways, 1 for uptrend).
        :param entry_price: The entry price of the trade.
        :param atr: The average true range (ATR) value.
        :param max_price: The maximum price since entry.
        :param atr_multiplier: The ATR multiplier for the stop loss.
        :param trail_percent: The trailing stop percentage.
        :param stop_loss_percent: The static stop loss percentage.
        :return: The stop loss price.
        """
        current_price = kwargs.get("current_price", None)
        atr = kwargs.get("atr", None)
        max_price = kwargs.get("max_price", None)
        entry_price = kwargs.get("entry_price", None)
        if current_price is None or atr is None or max_price is None or entry_price is None:
            return -1
        if trend_dir == -1:
            stop_loss = StopLoss.atr_stop_loss(current_price, atr, atr_multiplier)
        elif trend_dir == 1: 
            stop_loss = StopLoss.trailing_stop_loss(max_price, trail_percent)
        else: 
            stop_loss = StopLoss.static_stop_loss(entry_price, stop_loss_percent)
        return max(stop_loss,0)


class TakeProfit:


    @staticmethod
    def risk_reward_take_profit(risk_reward_ratio=2, **kwargs):
        """
        Calculate take-profit price based on risk-reward ratio.

        :param entry_price: Entry pårice of the trade.
        :param stop_loss_price: Stop loss price of the trade.
        :param risk_reward_ratio: Desired risk-reward ratio (e.g., 2 for a 2:1 ratio).
        :return: Take-profit price.
        """
        entry_price = kwargs.get("entry_price", None)
        stop_loss_price = kwargs.get("stop_loss_price", None)
        if entry_price is None or stop_loss_price is None:
            return -1
        risk_amount = entry_price - stop_loss_price
        take_profit_price = entry_price + (risk_amount * risk_reward_ratio)
        
        return take_profit_price

    @staticmethod
    def trailing_take_profit(trail_percent=0.05, **kwargs):
        """
        Calculate trailing take-profit price based on percentage.

        :param entry_price: Entry price of the trade.
        :param max_price: Highest price reached since entry.
        :param trail_percent: Percentage for trailing stop (e.g., 0.05 for 5%).
        :return: Trailing take-profit price.
        """
        # Validate inputs
        entry_price = kwargs.get("entry_price", None)
        max_price = kwargs.get("max_price", None)

        if entry_price <= 0 or max_price <= 0 or trail_percent <= 0:
            return -1
        if max_price < entry_price:
            return -1
        trail_amount = max_price * trail_percent
        trailing_take_profit_price = max_price - trail_amount
        
        return trailing_take_profit_price

    @staticmethod
    def static_take_profit(take_profit_percent, **kwargs):
        """
        Calculate the take-profit price based on a fixed percentage above the entry price.

        :param entry_price: Entry price of the trade.
        :param take_profit_percent: Percentage above the entry price for taking profit.
        :return: Take-profit price.
        """
        entry_price = kwargs.get("entry_price", None)
        take_profit_price = entry_price * (1 + take_profit_percent)
        return take_profit_price

class PositionSizing:
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """
        Calculate the Kelly Criterion fraction for position sizing.

        :param win_rate: Probability of winning a trade (e.g., 0.6 for 60%).
        :param avg_win: Average win amount in dollars.
        :param avg_loss: Average loss amount in dollars.
        :return: Kelly fraction (fraction of capital to risk).
        """
        kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0, kelly_fraction)  # Ensure non-negative value
    
    @staticmethod
    def risk_based_position_size(capital, risk_per_trade, current_price, stop_loss_price):
        """
        Calculate the position size based on risk per trade and stop loss distance.

        :param capital: Total trading capital in dollars.
        :param risk_per_trade: Fraction of capital to risk on a single trade (e.g., 0.01 for 1%).
        :param entry_price: Entry price of the trade in dollars.
        :param stop_loss_price: Stop loss price of the trade in dollars.
        :return: Position size (number of shares to trade).
        """
        stop_loss_distance = abs(current_price - stop_loss_price)/current_price
        if stop_loss_distance <= 1e-6:
            return -1
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_loss_distance
        if position_size > capital: # Ensure position size doesn't exceed capital
            return capital / current_price
        num_shares = position_size / current_price
        return num_shares
    
    @staticmethod
    def fixed_fractional_position_size(capital, risk_percentage, current_price, stop_loss_price):
        """
        Calculate the position size using fixed fractional position sizing.

        :param capital: Total trading capital in dollars.
        :param risk_percentage: Percentage of capital to risk on a single trade.
        :param entry_price: Entry price of the trade in dollars.
        :param stop_loss_price: Stop loss price of the trade in dollars.
        :return: Position size (number of shares to trade).
        """
        risk_amount = capital * (risk_percentage / 100)
        risk_per_share = abs(current_price - stop_loss_price)
        position_size = risk_amount / risk_per_share
        if position_size > capital: # Ensure position size doesn't exceed capital
            return capital / current_price    
        num_shares = position_size / current_price
        return num_shares

