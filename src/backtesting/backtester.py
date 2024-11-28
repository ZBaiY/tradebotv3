import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.historical_data_handler import HistoricalDataHandler
from src.portfolio_management.risk_manager import RiskManager
from src.portfolio_management.single_risk import SingleRiskManager
from src.strategy.multi_asset_strategy import MultiAssetStrategy

from src.strategy.single_asset_strategy import SingleAssetStrategy
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor
from src.models.base_model import ForTesting as TestModel
from src.portfolio_management.portfolio_manager import PortfolioManager
from src.backtesting.performance_evaluation import MultiAssetPerformanceEvaluator, SingleAssetPerformanceEvaluator
from src.live_trading.order_manager import OrderManager

import pandas as pd

class SingleAssetBacktester:
    """
    A backtester for single-asset trading strategies.
    """
    def __init__(self, strategy: SingleAssetStrategy, data_handler: HistoricalDataHandler,
                 portfolio_manager: PortfolioManager, risk_manager: SingleRiskManager,
                order_manager: OrderManager,initial_capital: float = 100000.0):
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
        self.strategy = strategy
        self.data_handler = data_handler
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
        Runs the backtest for the single asset over the specified date range.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        print("Initializing single-asset backtest...")
        self.data_handler.load_data(start_date, end_date)
        self.current_date = start_date

        while not self.data_handler.is_end_of_data():
            self.current_date = self.data_handler.current_date()
            print(f"Processing date: {self.current_date}")

            market_data = self.data_handler.get_latest_data()
            signal = self.strategy.generate_signal(market_data)
            allocation = self.portfolio_manager.update_portfolio(
                signal, market_data, self.current_date
            )
            risk_adjusted_allocation = self.risk_manager.adjust_position(allocation)
            trade = self.order_manager.execute_order(risk_adjusted_allocation, market_data)

            # Log trade and update portfolio value
            if trade:
                self.trade_log.append(trade)
            self.portfolio_value = self.portfolio_manager.get_portfolio_value()

        print("Single-asset backtest completed.")

    def evaluate_performance(self):
        """
        Evaluates the performance of the strategy.

        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        from backtesting.performance_evaluation import SingleAssetPerformanceEvaluator
        performance_eval = SingleAssetPerformanceEvaluator(self.trade_log, self.initial_capital)
        metrics = performance_eval.calculate_metrics()
        return metrics

    def save_results(self, output_path: str = "single_asset_backtest_results.csv"):
        """
        Saves the backtest results to a CSV file.

        Args:
            output_path (str): File path to save the results.
        """
        results_df = pd.DataFrame(self.trade_log)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}.")

class MultiAssetBacktester:
    def __init__(self, strategy: MultiAssetStrategy, data_handler: HistoricalDataHandler,
                 portfolio_manager: PortfolioManager, risk_manager: RiskManager,
                 order_manager: OrderManager,initial_capital: float = 100000.0):
        
        self.strategy = strategy
        self.data_handler = data_handler
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