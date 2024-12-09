import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pandas as pd

########
# Only in this module, some percentage is from 0 to 100, to be consistent with the financial domain.
# In other modules, the percentage is from 0 to 1.
########


class SingleAssetModelPerformanceEvaluator:
    """
    Evaluates performance metrics for single-asset trading strategies.
    """

    def __init__(self, trade_log, balance_history, initial_balance):
        """
        Initializes the evaluator.

        Args:
            trade_log (list): List of trade dictionaries.
            balance_history (list): Balance history during backtesting.
            initial_balance (float): Initial capital for the backtest.
        """
        self.trade_log = trade_log
        self.balance_history = balance_history
        self.initial_balance = initial_balance

    def calculate_metrics(self):
        """
        Calculates and returns all performance metrics.

        Returns:
            dict: Performance metrics including ROI, Sharpe Ratio, Max Drawdown, and more.
        """
        roi = self.get_roi()
        max_drawdown = self.max_drawdown_percentage()
        sharpe_ratio = self.get_sharpe_ratio()
        win_rate = self.get_win_rate()
        profit_factor = self.get_profit_factor()
        avg_trade_return = self.get_avg_trade_return()
        avg_win, avg_loss = self.get_avg_win_loss()

        return {
            'ROI (%)': roi,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Trade Return (%)': avg_trade_return,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss,
        }
    def find_returns(self):
        """
        Calculates the return for each trade in the trade log and updates the log.

        The return is calculated as:
            (exit_price - entry_price) / entry_price * 100 for buy trades.
            (entry_price - exit_price) / entry_price * 100 for sell trades.
        """
        previous_trade = None
        for trade in self.trade_log:
            if trade['order'] == 'buy':
                # Record this as the entry trade
                previous_trade = trade
                trade['return'] = 0  # No return for the opening of a position
            elif trade['order'] == 'sell' and previous_trade and previous_trade['order'] == 'buy':
                # Calculate return for the closing of a buy position
                entry_price = previous_trade['price']
                exit_price = trade['price']
                trade_return = ((exit_price - entry_price) / entry_price) * 100
                trade['return'] = trade_return
                previous_trade['return'] = trade_return
                previous_trade = None  # Reset previous trade after closing position
            else:
                trade['return'] = 0  # No meaningful return for other orders


    def get_roi(self):
        net_profit = self.balance_history[-1] - self.initial_balance
        return (net_profit / self.initial_balance) * 100

    def max_drawdown_percentage(self):
        balance_history = np.array(self.balance_history)
        max_accumulated_balance = np.maximum.accumulate(balance_history)
        drawdown = max_accumulated_balance - balance_history
        max_drawdown = np.max(drawdown)
        return (max_drawdown / np.max(max_accumulated_balance)) * 100

    def get_sharpe_ratio(self, risk_free_rate=0.02):
        returns = np.diff(self.balance_history) / np.array(self.balance_history[:-1])
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        return (mean_returns - risk_free_rate) / std_returns if std_returns > 0 else np.nan

    def get_win_rate(self):
        win_trades = sum(1 for trade in self.trade_log if trade.get('return', 0) > 0)
        total_trades = len(self.trade_log)
        return (win_trades / total_trades) * 100 if total_trades > 0 else 0

    def get_profit_factor(self):
        gross_profit = sum(trade['return'] for trade in self.trade_log if trade.get('return', 0) > 0)
        gross_loss = abs(sum(trade['return'] for trade in self.trade_log if trade.get('return', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf

    def get_avg_trade_return(self):
        avg_return = np.mean([trade['return'] for trade in self.trade_log if 'return' in trade])
        return avg_return * 100

    def get_avg_win_loss(self):
        profits = [trade['return'] for trade in self.trade_log if trade.get('return', 0) > 0]
        losses = [trade['return'] for trade in self.trade_log if trade.get('return', 0) < 0]
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        return avg_win * 100, avg_loss * 100




class MultiAssetPerformanceEvaluator:
    """
    Evaluates performance metrics for multi-asset trading strategies.
    """

    def __init__(self, trade_log, portfolio_value_history, initial_balance):
        """
        Initializes the evaluator.

        Args:
            trade_log (list): List of trade dictionaries.
            portfolio_value_history (list): Portfolio value history during backtesting.
            initial_balance (float): Initial capital for the backtest.
        """
        self.trade_log = trade_log
        self.portfolio_value_history = portfolio_value_history
        self.initial_balance = initial_balance

    def calculate_metrics(self):
        """
        Calculates and returns all performance metrics.

        Returns:
            dict: Performance metrics including ROI, Sharpe Ratio, Max Drawdown, and more.
        """
        roi = self.get_roi()
        max_drawdown = self.max_drawdown_percentage()
        sharpe_ratio = self.get_sharpe_ratio()
        win_rate = self.get_win_rate()
        profit_factor = self.get_profit_factor()
        avg_trade_return = self.get_avg_trade_return()
        avg_win, avg_loss = self.get_avg_win_loss()

        return {
            'ROI (%)': roi,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Trade Return (%)': avg_trade_return,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss,
        }

    def get_roi(self):
        net_profit = self.portfolio_value_history[-1] - self.initial_balance
        return (net_profit / self.initial_balance) * 100

    def max_drawdown_percentage(self):
        portfolio_history = np.array(self.portfolio_value_history)
        max_accumulated_portfolio = np.maximum.accumulate(portfolio_history)
        drawdown = max_accumulated_portfolio - portfolio_history
        max_drawdown = np.max(drawdown)
        return (max_drawdown / np.max(max_accumulated_portfolio)) * 100

    def get_sharpe_ratio(self, risk_free_rate=0.02):
        returns = np.diff(self.portfolio_value_history) / np.array(self.portfolio_value_history[:-1])
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        return (mean_returns - risk_free_rate) / std_returns if std_returns > 0 else np.nan

    def get_win_rate(self):
        win_trades = sum(1 for trade in self.trade_log if trade.get('return', 0) > 0)
        total_trades = len(self.trade_log)
        return (win_trades / total_trades) * 100 if total_trades > 0 else 0

    def get_profit_factor(self):
        gross_profit = sum(trade['return'] for trade in self.trade_log if trade.get('return', 0) > 0)
        gross_loss = abs(sum(trade['return'] for trade in self.trade_log if trade.get('return', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf

    def get_avg_trade_return(self):
        avg_return = np.mean([trade['return'] for trade in self.trade_log if 'return' in trade])
        return avg_return * 100

    def get_avg_win_loss(self):
        profits = [trade['return'] for trade in self.trade_log if trade.get('return', 0) > 0]
        losses = [trade['return'] for trade in self.trade_log if trade.get('return', 0) < 0]
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        return avg_win * 100, avg_loss * 100





class PerformanceEvaluator:
    """
    A class to calculate various performance metrics for trading strategies.
    """

    @staticmethod
    def get_roi(net_profit, initial_balance):
        return (net_profit / initial_balance) * 100

    @staticmethod
    def get_max_drawdown(balance_history):
        balance_history = np.array(balance_history)
        return np.max(np.maximum.accumulate(balance_history) - balance_history)

    @staticmethod
    def max_drawdown_percentage(balance_history):
        balance_history = np.array(balance_history)
        max_accumulated_balance = np.maximum.accumulate(balance_history)
        drawdown = max_accumulated_balance - balance_history
        max_drawdown = np.max(drawdown)
        max_drawdown_percentage = (max_drawdown / np.max(max_accumulated_balance)) * 100
        return max_drawdown_percentage

    @staticmethod
    def get_sharpe_ratio(balance_history, risk_free_rate=0.02):
        with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warnings
            returns = np.where(balance_history[:-1] != 0, np.diff(balance_history) / balance_history[:-1], np.nan)
        mean_returns = np.nanmean(returns)
        std_returns = np.nanstd(returns)
        sharpe_ratio = (mean_returns - risk_free_rate) / std_returns if std_returns != 0 else np.nan
        return sharpe_ratio

    @staticmethod
    def get_win_rate(trade_log):
        win_trades = sum(1 for trade in trade_log if trade.get('return', 0) > 0)
        total_trades = len(trade_log)
        return (win_trades / total_trades) * 100 if total_trades > 0 else 0

    @staticmethod
    def get_profit_factor(trade_log):
        gross_profit = sum(trade['return'] for trade in trade_log if trade.get('return', 0) > 0)
        gross_loss = abs(sum(trade['return'] for trade in trade_log if trade.get('return', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else np.inf

    @staticmethod
    def get_avg_trade_return(trade_log):
        total_trades = len(trade_log)
        avg_return = np.mean([trade['return'] for trade in trade_log if 'return' in trade]) if total_trades > 0 else 0
        return avg_return * 100  # Convert to percentage

    @staticmethod
    def get_trade_frequency(data, trade_log):
        total_days = (data.index[-1] - data.index[0]).days
        total_trades = len(trade_log)
        return total_trades / total_days if total_days > 0 else 0

    @staticmethod
    def get_avg_win_loss(trade_log):
        profits = [trade['return'] for trade in trade_log if trade.get('return', 0) > 0]
        losses = [trade['return'] for trade in trade_log if trade.get('return', 0) < 0]
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        return avg_win * 100, avg_loss * 100  # Convert to percentage

    @staticmethod
    def calculate_performance(data, signals, initial_balance=10000.0):
        """
        Calculate key performance metrics for a trading strategy based on input signals.

        Args:
            data (pd.DataFrame): Historical price data with columns ['open_time', 'open', 'high', 'low', 'close', 'volume'].
            signals (pd.Series): Trading signals (1 for buy, -1 for sell, 0 for hold), aligned with the data.
            initial_balance (float): Starting balance for the backtest.

        Returns:
            dict: Calculated performance metrics.
        """
        # Ensure data and signals are aligned
        assert len(data) == len(signals), "Data and signals must have the same length"

        # Initialize variables
        balance = initial_balance
        position = 0
        trade_log = []
        balance_history = [balance]

        for i in range(1, len(data)):
            signal = signals.iloc[i]
            price = data['close'].iloc[i]

            if signal == "buy" and position == 0:
                position = balance / price  # Use all balance to buy
                balance = 0
                trade_log.append({'entry_time': data.index[i], 'entry_price': price, 'signal': 'buy'})

            elif signal == "sell" and position > 0:
                balance = position * price  # Sell everything
                position = 0
                trade_log[-1].update({
                    'exit_time': data.index[i],
                    'exit_price': price,
                    'return': (price - trade_log[-1]['entry_price']) / trade_log[-1]['entry_price']
                })

            balance_history.append(balance + position * price)

        # Final position liquidation
        if position > 0:
            balance = position * data['close'].iloc[-1]

        # Calculate performance metrics
        final_balance = balance
        net_profit = final_balance - initial_balance
        roi = PerformanceEvaluator.get_roi(net_profit, initial_balance)
        max_drawdown = PerformanceEvaluator.max_drawdown_percentage(balance_history)
        sharpe_ratio = PerformanceEvaluator.get_sharpe_ratio(balance_history)
        win_rate = PerformanceEvaluator.get_win_rate(trade_log)
        profit_factor = PerformanceEvaluator.get_profit_factor(trade_log)
        avg_trade_return = PerformanceEvaluator.get_avg_trade_return(trade_log)
        trade_frequency = PerformanceEvaluator.get_trade_frequency(data, trade_log)
        avg_win, avg_loss = PerformanceEvaluator.get_avg_win_loss(trade_log)

        return {
            'ROI (%)': roi,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Average Trade Return (%)': avg_trade_return,
            'Trade Frequency (trades/day)': trade_frequency,
            'Final Balance': final_balance,
            'Net Profit': net_profit,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss
        }
