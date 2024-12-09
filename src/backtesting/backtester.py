import os
import sys
from typing import List, Union
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.historical_data_handler import HistoricalDataHandler, SingleSymbolDataHandler, MultiSymbolDataHandler
from src.portfolio_management.risk_manager import RiskManager
from src.portfolio_management.single_risk import SingleRiskManager
from src.strategy.multi_asset_strategy import MultiAssetStrategy
from src.feature_engineering.feature_extractor import FeatureExtractor, SingleSymbolFeatureExtractor

from src.strategy.single_asset_strategy import SingleAssetStrategy
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor, MemSymbolProcessor
from src.models.base_model import ForTesting as TestModel
from src.portfolio_management.portfolio_manager import PortfolioManager
from src.backtesting.model_evaluation import MultiAssetPerformanceEvaluator, SingleAssetModelPerformanceEvaluator
from src.backtesting.strategy_evaluation import SingleAssetStrategyEvaluator
from src.live_trading.order_manager import OrderManager

import pandas as pd
import json
import numpy as np
import tqdm

class SingleAssetBacktester:
    """
    A backtester for single-asset trading strategies.
    """
    def __init__(self, strategy: SingleAssetStrategy = None, data_handler: SingleSymbolDataHandler = None,
                 feature_handler: SingleSymbolFeatureExtractor = None, signal_processors: List[Union[NonMemSymbolProcessor, MemSymbolProcessor]] = [],
                 risk_manager: SingleRiskManager = None, order_manager: OrderManager = None, initial_capital: float = 100000.0):
        """
        Initializes the SingleAssetBacktester.

        Args:
            strategy (SingleAssetStrategy): The strategy to test.
            data_handler (HistoricalDataHandler): Module to fetch historical market data.
            portfolio_manager (PortfolioManager): Manages portfolio and asset allocation.
            risk_manager (RiskManager): Applies risk control mechanisms.
            execution_handler (ExecutionHandler): Simulates trade execution.
            initial_capital (float): Starting capital for the backtest.
        """
        self.s_config = json.load(open('backtest/single_strategy.json'))

        self.symbol = list(self.s_config.keys())[0]
        self.data_handler = data_handler if data_handler else SingleSymbolDataHandler(self.symbol)
        self.data_handler_copy = self.data_handler.copy()
        self.feature_handler = feature_handler if feature_handler else FeatureExtractor(self.symbol, self.data_handler)
        self.realtime_settings = json.load(open('config/fetch_real_time.json'))
        self.window_size = self.realtime_settings['memory_stting']['window_size']
        self.strategy = strategy
        self.model_category = None
        self.model_variant = None

        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.signal_processors = signal_processors
        self.order_manager = order_manager
        # Internal tracking
        self.current_date = None
        self.balance = 0 # asset balance, in USDT, for future development, only track the quantity of the asset
        self.trade_log = []
        self.equity_history = []
        self.balance_history = []
        self.asset_quantity = 0


        self.equity_full_position = initial_capital
        self.asset_full_position = []
        self.capital_full_position = []
        self.log_model = []
        self.performance_metrics_model = None
        self.performance_metrics_strategy = None

    def recalculate_balance(self, price): ### this redundancy is due to a design flaw in the risk manager
        self.balance = self.asset_quantity * price
        self.equity = self.balance + self.balance

    def calculate_eq_bal(self, price, quantity=0): # quantity positive for buy, negative for sell, 0 for no trade
        USDT_balance = self.equity - self.balance
        self.balance += quantity * price
        self.asset_quantity += quantity
        USDT_balance -= quantity * price
        self.equity = self.balance + USDT_balance


    def equity_balance(self):
        self.risk_manager.set_equity(self.equity)
        self.strategy.set_equity(self.equity)
        self.risk_manager.set_balance(self.balance)
        self.strategy.set_balance(self.balance)

    def run_initialization(self):
        self.equity_balance()
        self.initialize_Strategy(self.equity, self.balance, self.data_handler_copy)

    def initialize_Strategy(self, equity, balance, data_handler_copy):
        
        m_config = {}
        r_config = {}
        d_config = {}
        m_config = self.s_config[self.symbol]['model']
        r_config = self.s_config[self.symbol]['risk_manager']
        d_config = self.s_config[self.symbol]['decision_maker']
        assigned_percentage = 1
        self.model_category = m_config['method'].split('_')[0]
        self.model_variant = m_config['method'].split('_')[1]

        self.risk_manager = SingleRiskManager(self.symbol, equity, balance, assigned_percentage, r_config, data_handler_copy, self.signal_processors, self.feature_handler)
        self.strategy = SingleAssetStrategy(self.symbol, m_config, d_config, self.risk_manager, data_handler_copy, self.signal_processors, self.feature_handler)

    def run_backtest(self, start_date: str, end_date: str):
        """
        Runs the backtest for the single asset over the specified date range.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        print("Initializing single-asset backtest...")
        self.data_handler.load_data(start_date, end_date)
        self.current_date = start_date

        self.balance_history.append(self.balance) 
        self.equity_history.append(self.equity)
        self.capital_full_position.append(self.equity)
        self.asset_full_position.append(self.asset_quantity)

        backtest_len = len(self.data_handler.cleaned_data)
        i = 0
        start_date = self.data_handler.cleaned_data.index[0]
        end_date = self.data_handler.cleaned_data.index[-1]
        print(f"Starting backtest from {start_date} to {end_date}.")
        while i <= backtest_len:
            start_index = max(0, i - self.window_size)
            self.current_date = self.data_handler.cleaned_data.index[i]
            self.data_handler_copy.cleaned_data = self.data_handler.get_data_range(start_index, i)
            price = self.data_handler_copy.cleaned_data['close'].iloc[-1]  
            self.recalculate_balance(price)  ### this redundancy is due to a design flaw in the risk manager
            market_order = self.strategy.run_strategy_market(self.data_handler_copy.cleaned_data)
            market_order['price'] = price

            
            if market_order['signal'] == 'hold':
                i += 1
                self.balance_history.append(self.balance) 
                self.equity_history.append(self.equity)
                self.capital_full_position.append(self.equity)
                self.asset_full_position.append(self.asset_quantity)
                continue
            else:
                log_instance, model_backtest = self.execute_order(market_order)
                
                if log_instance['order'] == 'buy':
                    self.asset_full_position = self.equity/price
                    self.equity_full_position = 0
                elif log_instance['order'] == 'sell':
                    self.equity_full_position = self.asset_full_position * price
                    self.balance_full_position = 0

                self.balance_history.append(self.balance) 
                self.equity_history.append(self.equity)
                self.capital_full_position.append(self.equity)
                self.asset_full_position.append(self.asset_quantity)

                self.trade_log.append(log_instance) 
                self.log_model.append(model_backtest)
            i += 1

        print("Single-asset backtest completed. Evaluating performance...")
        self.performance_metrics_model = self.evaluate_performance_model()
        self.save_metrics_model




    def execute_order(self, market_decision):
        order = market_decision['signal']
        price = market_decision['price']
        quantity = market_decision['amount']
        if order == 'buy':
            quantity = abs(quantity)
        elif order == 'sell':
            quantity = -abs(quantity)
        self.calculate_eq_bal(price, quantity)
        self.equity_balance()
        return {'symbol': self.symbol, 'date': self.current_date, 'price': price, 'quantity': quantity, 'order': order, 'balance': self.balance, 'equity': self.equity}, {'symbol': self.symbol, 'date': self.current_date, 'price': price,'order': order}


    def evaluate_performance_model(self):
        """
        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        performance_eval = SingleAssetModelPerformanceEvaluator(self.log_model, self.capital_full_position, self.initial_capital)
        metrics = performance_eval.calculate_metrics()
        return metrics
    
    def evaluate_performance_strategy(self):
        """
        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        performance_eval = SingleAssetStrategyEvaluator(self.trade_log, self.equity_history, self.balance_history, self.initial_capital)
        metrics = performance_eval.calculate_metrics()
        return metrics
    
    def save_metrics_model(self, output_path: str = "backtest/performance/model"):
        file_name = f"{self.symbol}_{self.model_category}_{self.model_variant}.json"
        output_path = os.path.join(output_path, file_name)
        with open(output_path, 'w') as f:
            json.dump(self.performance_metrics_model, f)

        print(f"Results saved to {output_path}.")

    def save_metrics_strategy(self, output_path: str = "backtest/performance/strategy"):
        strategy = self.strategy.__class__.__name__
        variation = 'v1'
        file_name = f"{self.symbol}_{strategy}_{variation}.json"
        output_path = os.path.join(output_path, file_name)
        with open(output_path, 'w') as f:
            json.dump(self.performance_metrics_strategy, f)

        print(f"Results saved to {output_path}.")

class MultiAssetBacktester:
    def __init__(self, strategy: MultiAssetStrategy = None, data_handler: MultiSymbolDataHandler = None,
                 feature_handler: FeatureExtractor = None, signal_processors: List[NonMemSignalProcessor, SignalProcessor] = [],
                 portfolio_manager: PortfolioManager = None, risk_manager: RiskManager = None,
                 order_manager: OrderManager = None,initial_capital: float = 100000.0):
        
        self.s_config = json.load(open('backtest/strategy.json'))

        self.initial_capital = initial_capital
        self.total_equity = initial_capital
        self.symbols = list(self.s_config.keys())

        self.data_handler = data_handler if data_handler else MultiSymbolDataHandler(self.symbols)
        self.feature_handler = feature_handler
        self.signal_processors = signal_processors
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.strategy = strategy

        # Tracking balances and orders
        self.balances_symbol = {symbol: 0.0 for symbol in self.symbols}  # Balance for each symbol
        self.balances_usdt = 0.0  # USDT balance
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.equity_history = []
        self.balance_history = {symbol: [] for symbol in self.symbols}
        self.trade_logs = {symbol: [] for symbol in self.symbols}


    def equity_balance(self):
        """
        Calculate and update equity based on current balances and prices.
        """
        self.balances_usdt = self.order_manager.get_account_balance("USDT")  # Mock for backtesting
        total_equity = self.balances_usdt
        for symbol in self.symbols:
            price = self.data_handler.get_symbol_last_data(symbol)['close']
            total_equity += self.balances_symbol[symbol] * price
        self.total_equity = total_equity
        return total_equity

    def set_entry_prices(self):
        """
        Set entry prices for each symbol.
        """
        for symbol in self.symbols:
            self.entry_prices[symbol] = self.entry_price(symbol)
        self.risk_manager.set_entry_price(self.entry_prices)

    def entry_price(self, symbol):
        """
        Calculate the weighted average entry price for the current open position of a symbol.
        """
        past_trades = self.order_manager.fetch_past_trades_from_api(symbol)  # Mocked for backtesting
        total_qty, total_cost = 0.0, 0.0
        for trade in past_trades:
            trade_qty = float(trade["executedQty"])
            trade_price = float(trade["price"])
            total_qty += trade_qty
            total_cost += trade_qty * trade_price
        return total_cost / total_qty if total_qty > 0 else 0.0

    def run_backtest(self, start_date: str, end_date: str):
        """
        Run the backtest for all symbols within the specified date range.
        """
        print(f"Starting backtest from {start_date} to {end_date}.")

        # Load data for symbols
        self.data_handler.load_data(begin_date=start_date, end_date=end_date)
        self.equity_balance()

        # Backtesting loop
        for current_date in self.data_handler.symbol_handlers[self.symbols[0]].get_data_range(0, -1).index:  # Assuming all symbols have same date range
            for symbol in self.symbols:
                self.data_handler.get_symbol_data(symbol)  # Fetch latest data

                # Generate signals from strategy
                market_orders = self.strategy.run_strategy_market()

                # Process market orders
                for symbol, order in market_orders.items():
                    if order["signal"] == "hold":
                        continue
                    self.execute_order(symbol, order)

            # Update equity and balances
            self.equity_balance()
            self.log_state(current_date)

        print("Backtest completed. Calculating performance.")
        return self.evaluate_performance()

    def execute_order(self, symbol, order):
        """
        Execute buy/sell orders for a given symbol.
        """
        order_type = order["signal"]
        amount = order["amount"]
        price = self.data_handler.get_symbol_last_data(symbol)['close']

        if order_type == "buy":
            cost = amount * price
            if self.balances_usdt >= cost:
                self.balances_usdt -= cost
                self.balances_symbol[symbol] += amount
        elif order_type == "sell":
            if self.balances_symbol[symbol] >= amount:
                self.balances_usdt += amount * price
                self.balances_symbol[symbol] -= amount

        trade_log = {
            "symbol": symbol,
            "type": order_type,
            "amount": amount,
            "price": price,
            "date": self.data_handler.get_symbol_last_data(symbol).name
        }
        self.trade_logs[symbol].append(trade_log)

    def log_state(self, current_date):
        """
        Log the current state of equity and balances.
        """
        self.equity_history.append({
            "date": current_date,
            "total_equity": self.total_equity
        })
        for symbol in self.symbols:
            self.balance_history[symbol].append({
                "date": current_date,
                "balance": self.balances_symbol[symbol]
            })

    def evaluate_performance(self):
        """
        Evaluate the performance of the backtest.
        """
        performance = {}
        for symbol in self.symbols:
            performance[symbol] = {
                "total_trades": len(self.trade_logs[symbol]),
                "final_balance": self.balances_symbol[symbol],
                "total_equity": self.total_equity,
            }
        return performance

    def save_metrics(self, output_path="backtest_results/"):
        """
        Save performance metrics to JSON files.
        """
        os.makedirs(output_path, exist_ok=True)
        for symbol, metrics in self.evaluate_performance().items():
            file_name = f"{symbol}_performance.json"
            file_path = os.path.join(output_path, file_name)
            with open(file_path, "w") as file:
                json.dump(metrics, file)
            print(f"Saved performance metrics for {symbol} to {file_path}.")

            
if __name__ == "__main__":
    # Example of initializing and running the Backtester
    strategy = ...  # Replace with your custom strategy
    data_handler = HistoricalDataHandler()
    portfolio_manager = PortfolioManager()
    risk_manager = RiskManager()


    backtester = MultiAssetBacktester(
        strategy=strategy,
        data_handler=data_handler,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        initial_capital=100000.0
    )
    backtester.run_backtest("2022-01-01", "2023-01-01")
    performance = backtester.evaluate_performance()
    print("Performance Metrics:", performance)
    backtester.save_results()