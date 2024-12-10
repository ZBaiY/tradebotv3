import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd


class SingleAssetStrategyEvaluator:
    """
    A unified evaluator for trading strategies, covering returns, performance, and risk metrics.
    """

    def __init__(self, trade_log, equity_history, asset_balance_history, initial_balance):
        """
        Initializes the evaluator.

        Args:
            trade_log (list): List of trade dictionaries with details about trades.
            equity_history (list): Equity history during backtesting.
            asset_balance_history (list): Asset balance history (cash).
            initial_balance (float): Initial starting balance.
        """
        self.trade_log = trade_log
        self.equity_history = equity_history
        self.asset_balance_history = asset_balance_history
        self.initial_balance = initial_balance

    def calculate_trade_returns(self):
        """
        Calculates realized returns for each trade based on partial positions.
        """
        open_positions = {}
        for trade in self.trade_log:
            symbol = trade['symbol']
            price = trade['price']
            quantity = trade['quantity']
            order = trade['order']

            if order == 'buy':
                # Add to open positions
                if symbol not in open_positions:
                    open_positions[symbol] = []
                open_positions[symbol].append({'price': price, 'quantity': quantity})
                trade['return'] = 0  # No realized return for the opening trade

            elif order == 'sell':
                # Realize profits for the sold quantity
                if symbol in open_positions:
                    remaining_quantity = quantity
                    realized_profit = 0
                    while remaining_quantity > 0 and open_positions[symbol]:
                        entry = open_positions[symbol][0]
                        entry_quantity = entry['quantity']
                        if remaining_quantity >= entry_quantity:
                            profit = (price - entry['price']) * entry_quantity
                            realized_profit += profit
                            remaining_quantity -= entry_quantity
                            open_positions[symbol].pop(0)  # Remove fully used entry
                        else:
                            profit = (price - entry['price']) * remaining_quantity
                            realized_profit += profit
                            entry['quantity'] -= remaining_quantity
                            remaining_quantity = 0
                    trade['return'] = realized_profit / (quantity * price) * 100 if quantity > 0 else 0

    def calculate_metrics(self):
        """
        Combines performance and risk metrics into a single report.
        """
        self.calculate_trade_returns()  # Ensure trade returns are calculated first

        roi = self.get_roi()
        max_drawdown = self.max_drawdown_percentage()
        sharpe_ratio = self.get_sharpe_ratio()
        position_efficiency = self.get_position_efficiency()
        trade_utilization = self.get_trade_utilization()
        profit_attribution = self.get_profit_attribution()

        return {
            'ROI (%)': roi,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Position Efficiency (%)': position_efficiency,
            'Trade Utilization (%)': trade_utilization,
            'Profit Attribution': profit_attribution,
        }

    def get_roi(self):
        net_profit = self.equity_history[-1] - self.initial_balance
        return (net_profit / self.initial_balance) * 100

    def max_drawdown_percentage(self):
        equity_history = np.array(self.equity_history)
        max_accumulated_equity = np.maximum.accumulate(equity_history)
        drawdown = max_accumulated_equity - equity_history
        max_drawdown = np.max(drawdown)
        return (max_drawdown / np.max(max_accumulated_equity)) * 100

    def get_sharpe_ratio(self, risk_free_rate=0.02):
        returns = np.diff(self.equity_history) / np.array(self.equity_history[:-1])
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        return (mean_returns - risk_free_rate) / std_returns if std_returns > 0 else np.nan

    def get_position_efficiency(self):
        max_profit = sum(trade['return'] for trade in self.trade_log if trade.get('return', 0) > 0)
        actual_profit = sum(trade['return'] for trade in self.trade_log)
        return (actual_profit / max_profit) * 100 if max_profit > 0 else 0

    def get_trade_utilization(self):
        total_cash = sum(self.asset_balance_history)
        total_invested = sum(self.equity_history) - total_cash
        return (total_invested / total_cash) * 100 if total_cash > 0 else 0

    def get_profit_attribution(self):
        attribution = {}
        for trade in self.trade_log:
            symbol = trade['symbol']
            if symbol not in attribution:
                attribution[symbol] = 0
            attribution[symbol] += trade.get('return', 0)
        return attribution


class MultiSymbolStrategyEvaluator:
    """
    A unified evaluator for multi-symbol trading strategies, covering returns, performance, and risk metrics.
    """

    def __init__(self, trade_logs, equity_history, asset_balance_history, initial_balance):
        """
        Initializes the evaluator.

        Args:
            trade_logs (dict): Dictionary of trade logs per symbol.
            equity_history (list): Total equity history during backtesting.
            asset_balance_history (dict): Dictionary of asset balance histories per symbol.
            initial_balance (float): Initial starting balance.
        """
        self.trade_logs = trade_logs  # {symbol: [trades]}
        self.equity_history = equity_history  # Total equity history
        self.asset_balance_history = asset_balance_history  # {symbol: [balances]}
        self.initial_balance = initial_balance

    def calculate_trade_returns(self):
        """
        Calculates realized returns for each trade for all symbols.
        """
        open_positions = {symbol: [] for symbol in self.trade_logs}

        for symbol, trades in self.trade_logs.items():
            for trade in trades:
                price = trade['price']
                quantity = trade['quantity']
                order = trade['order']

                if order == 'buy':
                    # Add to open positions
                    open_positions[symbol].append({'price': price, 'quantity': quantity})
                    trade['return'] = 0  # No realized return for the opening trade

                elif order == 'sell':
                    # Realize profits for the sold quantity
                    remaining_quantity = quantity
                    realized_profit = 0
                    while remaining_quantity > 0 and open_positions[symbol]:
                        entry = open_positions[symbol][0]
                        entry_quantity = entry['quantity']
                        if remaining_quantity >= entry_quantity:
                            profit = (price - entry['price']) * entry_quantity
                            realized_profit += profit
                            remaining_quantity -= entry_quantity
                            open_positions[symbol].pop(0)  # Remove fully used entry
                        else:
                            profit = (price - entry['price']) * remaining_quantity
                            realized_profit += profit
                            entry['quantity'] -= remaining_quantity
                            remaining_quantity = 0
                    trade['return'] = realized_profit / (quantity * price) * 100 if quantity > 0 else 0

    def calculate_metrics(self):
        """
        Combines performance and risk metrics into a single report.
        """
        self.calculate_trade_returns()  # Ensure trade returns are calculated first

        roi = self.get_roi()
        max_drawdown = self.max_drawdown_percentage()
        sharpe_ratio = self.get_sharpe_ratio()
        symbol_roi = self.get_symbol_roi()
        trade_efficiency = self.get_trade_efficiency()
        trade_distribution = self.get_trade_distribution()
        profit_attribution = self.get_profit_attribution()

        return {
            'Total ROI (%)': roi,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Symbol ROI (%)': symbol_roi,
            'Trade Efficiency (%)': trade_efficiency,
            'Trade Distribution': trade_distribution,
            'Profit Attribution': profit_attribution,
        }

    def get_roi(self):
        net_profit = self.equity_history[-1] - self.initial_balance
        return (net_profit / self.initial_balance) * 100

    def max_drawdown_percentage(self):
        equity_history = np.array(self.equity_history)
        max_accumulated_equity = np.maximum.accumulate(equity_history)
        drawdown = max_accumulated_equity - equity_history
        max_drawdown = np.max(drawdown)
        return (max_drawdown / np.max(max_accumulated_equity)) * 100

    def get_sharpe_ratio(self, risk_free_rate=0.02):
        returns = np.diff(self.equity_history) / np.array(self.equity_history[:-1])
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        return (mean_returns - risk_free_rate) / std_returns if std_returns > 0 else np.nan

    def get_symbol_roi(self):
        """
        Calculates ROI for each symbol individually.
        """
        symbol_roi = {}
        for symbol, trades in self.trade_logs.items():
            total_cost = 0
            total_return = 0
            for trade in trades:
                if trade['order'] == 'buy':
                    total_cost += trade['price'] * trade['quantity']
                elif trade['order'] == 'sell':
                    total_return += trade['price'] * trade['quantity']
            symbol_roi[symbol] = (total_return - total_cost) / total_cost * 100 if total_cost > 0 else 0
        return symbol_roi

    def get_trade_efficiency(self):
        """
        Measures efficiency of trades in capturing potential profits across all symbols.
        """
        total_profit = 0
        max_possible_profit = 0

        for symbol, trades in self.trade_logs.items():
            for trade in trades:
                if trade.get('return', 0) > 0:
                    total_profit += trade['return']
                max_possible_profit += abs(trade.get('return', 0))  # Assuming full profit potential captured

        return (total_profit / max_possible_profit) * 100 if max_possible_profit > 0 else 0

    def get_trade_distribution(self):
        """
        Evaluates the distribution of trades among symbols.
        """
        distribution = {symbol: len(trades) for symbol, trades in self.trade_logs.items()}
        total_trades = sum(distribution.values())
        return {symbol: (count / total_trades) * 100 for symbol, count in distribution.items()} if total_trades > 0 else {}

    def get_profit_attribution(self):
        """
        Calculates profit attribution for each symbol.
        """
        attribution = {}
        for symbol, trades in self.trade_logs.items():
            for trade in trades:
                if symbol not in attribution:
                    attribution[symbol] = 0
                attribution[symbol] += trade.get('return', 0)
        return attribution
