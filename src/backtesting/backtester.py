import os
import sys
from typing import List, Union
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.historical_data_handler import HistoricalDataHandler, SingleSymbolDataHandler
from src.portfolio_management.risk_manager import RiskManager
from src.portfolio_management.single_risk import SingleRiskManager
from src.strategy.multi_asset_strategy import MultiAssetStrategy
from src.feature_engineering.feature_extractor import FeatureExtractor, SingleSymbolFeatureExtractor

from src.strategy.single_asset_strategy import SingleAssetStrategy
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor, MemSymbolProcessor
from src.models.base_model import ForTesting as TestModel
from src.portfolio_management.portfolio_manager import PortfolioManager
from src.backtesting.performance_evaluation import MultiAssetPerformanceEvaluator, SingleAssetPerformanceEvaluator
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
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.signal_processors = signal_processors
        self.order_manager = order_manager
        # Internal tracking
        self.current_date = None
        self.balance = 0 # asset balance, in USDT
        self.balance_full_position = 0
        self.equity_full_position = initial_capital
        self.balance_history = []
        self.balance_full_position = []
        self.equity_history = []
        self.trade_log = []
        self.log_model = []

    def calculate_eq_bal(self, price, quantity=0): # quantity positive for buy, negative for sell, 0 for no trade
        USDT_balance = self.equity - self.balance
        self.balance += quantity * price
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
        backtest_len = len(self.data_handler.cleaned_data)
        i = 0
        start_date = self.data_handler.cleaned_data.index[0]
        end_date = self.data_handler.cleaned_data.index[-1]
        print(f"Starting backtest from {start_date} to {end_date}.")
        while i <= backtest_len:
            start_index = max(0, i - self.window_size)
            self.current_date = self.data_handler.cleaned_data.index[i]
            self.data_handler_copy.cleaned_data = self.data_handler.get_data_range(start_index, i)
            market_order = self.strategy.run_strategy_market(self.data_handler_copy.cleaned_data)
            market_order['price'] = self.data_handler_copy.cleaned_data['close'].iloc[-1]
            
            if market_order['signal'] == 'hold':
                i += 1
                self.equity_history.append(self.equity)
                self.balance_history.append(self.balance)
                self.balance_full_position.append(self.balance)
                continue
            else:
                log_instance, model_backtest = self.execute_order(market_order)
                if log_instance['order'] == 'buy':
                    self.balance_full_position = self.equity
                    self.equity_full_position = 0
                self.trade_log.append(log_instance) 
                self.log_model.append(model_backtest)
                self.equity_history.append(self.equity)
                self.balance_history.append(self.balance)
            i += 1

        print("Single-asset backtest completed. Evaluating performance...")


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
        from backtesting.performance_evaluation import SingleAssetPerformanceEvaluator
        performance_eval = SingleAssetPerformanceEvaluator(self.log_model, self.equity_history, self.initial_capital)
        metrics = performance_eval.calculate_metrics()
        return metrics

    def save_results(self, output_path: str = "backtest/performance"):
        strategy = self.strategy.__class__.__name__
        variation = 'v1'
        file_name = f"{self.symbol}_{strategy}_{variation}.csv"
        output_path = os.path.join(output_path, file_name)
        results_df = pd.DataFrame(self.trade_log)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}.")

class MultiAssetBacktester:
    def __init__(self, strategy: MultiAssetStrategy = None, data_handler: HistoricalDataHandler = None,
                 feature_handler: FeatureExtractor = None, signal_processor: List[NonMemSignalProcessor, SignalProcessor] = [],
                 portfolio_manager: PortfolioManager = None, risk_manager: RiskManager = None,
                 order_manager: OrderManager = None,initial_capital: float = 100000.0):
        
        self.strategy = strategy
        self.data_handler = data_handler if data_handler else HistoricalDataHandler()
        self.feature_handler = feature_handler if feature_handler else FeatureExtractor(self.data_handler)
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.order_manager = order_manager

        # Internal tracking
        self.current_date = None
        self.portfolio_value = initial_capital
        self.trade_log = []

    def run_backtest(self, start_date: str, end_date: str):
        """
        Runs the backtest over the specified date range.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        print("Initializing backtest...")
        self.data_handler.load_data(start_date, end_date)
        self.current_date = start_date

        while not self.data_handler.is_end_of_data():
            self.current_date = self.data_handler.current_date()
            print(f"Processing date: {self.current_date}")

            market_data = self.data_handler.get_latest_data()
            signals = self.strategy.generate_signals(market_data)
            allocations = self.portfolio_manager.update_portfolio(
                signals, market_data, self.current_date
            )
            risk_adjusted_allocations = self.risk_manager.adjust_positions(allocations)

            # Step 5: Execute Trades
            trades = self.order_manager.execute_orders(risk_adjusted_allocations, market_data)
            # Log trades and update portfolio value
            self.trade_log.extend(trades)
            self.portfolio_value = self.portfolio_manager.get_portfolio_value()

        print("Backtest completed.")

    def evaluate_performance(self):
        """
        Evaluates the performance of the strategy.
        
        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        performance_eval = MultiAssetPerformanceEvaluator(self.trade_log, self.initial_capital)
        metrics = performance_eval.calculate_metrics()
        return metrics

    def save_results(self, output_path: str = "backtest_results.csv"):
        """
        Saves the backtest results to a CSV file.
        
        Args:
            output_path (str): File path to save the results.
        """
        results_df = pd.DataFrame(self.trade_log)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}.")

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