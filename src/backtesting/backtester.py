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
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor, NonMemSymbolProcessor, MemSymbolProcessor, NonMemSymbolProcessorDataSymbol
from src.models.base_model import ForTesting as TestModel
from src.portfolio_management.portfolio_manager import PortfolioManager
from src.backtesting.model_evaluation import  SingleAssetModelPerformanceEvaluator
from src.backtesting.strategy_evaluation import SingleAssetStrategyEvaluator, MultiSymbolStrategyEvaluator
from src.live_trading.order_manager import OrderManager

import pandas as pd
import json
import numpy as np
import gc
from tqdm import tqdm

fee = 0.001

class SingleAssetBacktester:
    """
    A backtester for single-asset trading strategies.
    """
    def __init__(self, strategy: SingleAssetStrategy = None, data_handler: SingleSymbolDataHandler = None,
                 feature_handler: SingleSymbolFeatureExtractor = None, signal_processors: List[Union[NonMemSymbolProcessorDataSymbol, MemSymbolProcessor]] = [],
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
        self.s_config = json.load(open('backtest/config/single_strategy.json'))
        self.symbol = list(self.s_config.keys())[0]
        self.interval_str = self.s_config[self.symbol]['interval']
        self.start_date = self.s_config[self.symbol]['start_date']
        self.end_date = self.s_config[self.symbol]['end_date']
        self.data_handler = data_handler if data_handler else SingleSymbolDataHandler(self.symbol)
        self.data_handler.set_dates(self.start_date, self.end_date)
        # print(self.data_handler.get_data().head())
        self.data_handler_copy = self.data_handler.copy()
        self.feature_handler = feature_handler if feature_handler else SingleSymbolFeatureExtractor(self.symbol, self.data_handler_copy)
        self.interval_str = self.data_handler.interval_str
        self.window_size = self.data_handler.window_size
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
        self.entry_price = -1 # initialize entry price to -1
        self.asset_quantity = 0


        self.USDT_full_position = initial_capital
        self.asset_full_position = []
        self.capital_full_position = []
        self.log_model = []
        self.performance_metrics_model = None
        self.performance_metrics_strategy = None

    def recalculate_balance(self, price): ### this redundancy is due to a design flaw in the risk manager
        USDT_balance = self.equity - self.balance
        self.balance = self.asset_quantity * price
        self.equity = USDT_balance + self.balance

    def calculate_eq_bal(self, price, quantity=0): # quantity positive for buy, negative for sell, 0 for no trade
        # print(self.equity)
        # input("backtestor 86, Press Enter to continue...")
        USDT_balance = self.equity - self.balance
        self.balance += quantity * price
        self.asset_quantity += quantity
        USDT_balance -= quantity * price + abs(quantity) * price * fee
        USDT_balance = max(0, USDT_balance) # might have rounding error
        self.equity = self.balance + USDT_balance



    def equity_balance(self):
        self.risk_manager.set_equity(self.equity)
        self.strategy.set_equity(self.equity)
        self.risk_manager.set_balance(self.balance)
        self.strategy.set_balance(self.balance)
        self.risk_manager.set_position(self.balance/self.equity)
        

    def run_initialization(self):
        self.initialize_Strategy(self.equity, self.balance, self.data_handler_copy)
        self.equity_balance()

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
        self.strategy.initialize(self.risk_manager)


    def run_backtest(self):
        """
        Runs the backtest for the single asset over the specified date range.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        print("Initializing single-asset backtest...")
        self.data_handler.load_data(interval_str=self.interval_str, begin_date=self.start_date, end_date=self.end_date)
        self.balance_history.append(self.balance) 
        self.equity_history.append(self.equity)
        self.capital_full_position.append(self.equity)
        self.asset_full_position.append(self.asset_quantity)
        i = 10*self.window_size
        backtest_len = len(self.data_handler.cleaned_data)-i
        start_date = self.data_handler.cleaned_data.index[0]
        end_date = self.data_handler.cleaned_data.index[-1]
        print(f"Starting backtest from {start_date} to {end_date}.")
        start_index = max(0, i - self.window_size)
        self.current_date = self.data_handler.cleaned_data.index[i]
        self.data_handler_copy.cleaned_data = self.data_handler.get_data_range(start_index, i)
        self.feature_handler.pre_run_indicators()
        quantity_full = 0
        for i in tqdm(range(i, len(self.data_handler.cleaned_data)-1)):
            i += 1
            start_index = max(0, i - self.window_size)
            self.current_date = self.data_handler.cleaned_data.index[i]
            self.data_handler_copy.cleaned_data = self.data_handler.get_data_range(start_index, i)
            self.data_handler_copy.cleaned_data = self.data_handler_copy.cleaned_data.tail(self.window_size)
            # To mimic the live trading environment, we only have the last data point
            
            price = self.data_handler_copy.cleaned_data.iloc[-1]['close']
            self.recalculate_balance(price)  ### this redundancy is due to a design flaw in the risk manager
            self.equity_balance()
            self.feature_handler.update(self.data_handler_copy.cleaned_data.iloc[[-1]])
            market_order = self.strategy.run_strategy_market()
            market_order['price'] = price
            

            if market_order['signal'] == 'hold':
                self.balance_history.append(self.balance) 
                self.equity_history.append(self.equity)
                self.capital_full_position.append(self.capital_full_position[-1])
                self.asset_full_position.append(self.asset_full_position[-1])
                continue
            else:
                log_instance, model_backtest = self.execute_order(market_order)
                
                if log_instance['order'] == 'buy' and self.USDT_full_position > 0:
                    quantity_full = self.USDT_full_position/price
                    self.USDT_full_position = 0
                    model_backtest['amount'] = quantity_full
                    self.log_model.append(model_backtest)
                    self.capital_full_position.append(quantity_full * price)
                elif log_instance['order'] == 'sell' and self.asset_full_position[-1] > 0:
                    model_backtest['amount'] = -self.asset_full_position[-1]
                    quantity_full = 0
                    self.USDT_full_position = self.asset_full_position[-1] * price
                    self.log_model.append(model_backtest)
                    self.capital_full_position.append(self.USDT_full_position)
                else:
                    # model_backtest['order'] = 'hold'
                    model_backtest['amount'] = 0
                    self.log_model.append(model_backtest)
                    if self.USDT_full_position > 0:
                        self.capital_full_position.append(self.USDT_full_position)
                    else:
                        self.capital_full_position.append(self.asset_full_position[-1] * price)
                self.balance_history.append(self.balance) 
                self.equity_history.append(self.equity)
                self.asset_full_position.append(quantity_full)


                self.trade_log.append(log_instance) 
            if i % 100 == 0:
                del log_instance, model_backtest, market_order
                gc.collect()
                # print(i, log_instance)
                # input("backtestor 183, Press Enter to continue...")

        print("Single-asset backtest completed. Evaluating performance...")
        
        self.performance_metrics_model = self.evaluate_performance_model()
        self.save_metrics_model()
        self.performance_metrics_strategy = self.evaluate_performance_strategy()
        self.save_metrics_strategy()


    def calculate_entry_price(self):
        """
        Calculate the weighted average entry price based on the asset quantity and trade log.
        
        :return: The average entry price, -1 if no position, or 0 if the position is fully closed.
        """
        total_cost = 0.0
        total_quantity = self.asset_quantity  # Start with the actual asset quantity
        threshold = 1e-6  # Small value to handle floating-point precision errors

        # If the asset quantity is None or too small, return -1 (no open position)
        if total_quantity is None or total_quantity < threshold:
            return -1

        # Process the trade log anti-chronologically (most recent trades first)
        for trade in reversed(self.trade_log):
            trade_qty = trade["quantity"]
            trade_price = trade["price"]
            trade_type = trade["order"]  # 'buy' or 'sell'

            if trade_type == "buy":
                total_cost += trade_qty * trade_price
                total_quantity -= trade_qty  # Adjust quantity needing reconciliation
            elif trade_type == "sell":
                total_quantity += trade_qty  # Selling increases the amount needing reconciliation

            # Stop early if we have reconciled the asset quantity
            if abs(total_quantity) < threshold:
                break

            # If total_quantity goes negative, log a warning and stop
            if total_quantity < 0:
                # print("Trade log inconsistency detected. Remaining quantity dropped below zero. Stopping calculation.")
                break

        # Calculate and return the weighted average entry price
        if total_quantity > 0:
            weighted_entry_price = total_cost / total_quantity
            return weighted_entry_price
        else:
            return 0.0


    def execute_order(self, market_decision):
        order = market_decision['signal']
        price = market_decision['price']
        quantity = market_decision['amount']
    
        if abs(quantity) <= 1e-6:
            order = 'hold'
            quantity = 0
        if order == 'buy':
            if quantity * price > (self.equity-self.balance)/(1+fee):
                quantity = (self.equity-self.balance)/price/(1+fee)
            quantity = abs(quantity)
        elif order == 'sell':
            if quantity * price > self.balance:
                quantity = self.balance/price
            quantity = -abs(quantity)
        # print('balance', self.balance, 'equity', self.equity, 'self quantity', self.asset_quantity, 'order quantity', quantity)
        # print('position size', self.risk_manager.position)
        self.calculate_eq_bal(price, quantity)
        self.equity_balance()
        return {'symbol': self.symbol, 
                'date': self.current_date, 
                'price': price, 
                'quantity': quantity, 
                'order': order, 
                'balance': self.balance, 
                'equity': self.equity}, {'symbol': self.symbol, 'date': self.current_date, 'price': price,'order': order}


    def evaluate_performance_model(self):
        """
        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        performance_eval = SingleAssetModelPerformanceEvaluator(self.log_model, self.capital_full_position, self.initial_capital, self.interval_str)
        metrics = performance_eval.calculate_metrics()
        return metrics
    
    def evaluate_performance_strategy(self):
        """
        Returns:
            dict: Performance metrics such as Sharpe Ratio, max drawdown, etc.
        """
        performance_eval = SingleAssetStrategyEvaluator(self.trade_log, self.equity_history, self.balance_history, self.initial_capital, self.interval_str)
        metrics = performance_eval.calculate_metrics()
        return metrics
    
    def save_metrics_model(self, output_path: str = "backtest/performance/model"):
        file_name = f"{self.symbol}_{self.model_category}_{self.model_variant}.json"
        output_path = os.path.join(output_path, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.performance_metrics_model, f, indent=4)  # Add indent=4 for proper formatting

        print(f"Results saved to {output_path}.")

    def save_metrics_strategy(self, output_path: str = "backtest/performance/strategy"):
        strategy = self.strategy.__class__.__name__
        variation = 'v1'
        file_name = f"{self.symbol}_{strategy}_{variation}.json"
        output_path = os.path.join(output_path, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.performance_metrics_strategy, f, indent=4)

        print(f"Results saved to {output_path}.")

class MultiAssetBacktester:
    def __init__(self, strategy: MultiAssetStrategy = None, data_handler: MultiSymbolDataHandler = None,
                 feature_handler: FeatureExtractor = None, signal_processors:  List[Union[NonMemSymbolProcessor, MemSymbolProcessor]] = [],
                 portfolio_manager: PortfolioManager = None, risk_manager: RiskManager = None,
                 order_manager: OrderManager = None,initial_capital: float = 100000.0):
        
        self.s_config = json.load(open('backtest/config/strategy.json'))
        self.start_date = self.s_config['date_range']['start_date']      
        self.end_date = self.s_config['date_range']['end_date']  
        self.symbols = list(self.s_config.keys())[1:]
        self.interval_str = self.s_config['date_range']['interval']

        self.data_handler = data_handler if data_handler else MultiSymbolDataHandler(self.symbols)
        self.window_size = self.data_handler.window_size
        self.data_handler.set_dates(self.start_date, self.end_date)
        self.data_handler.load_data(interval_str=self.interval_str, begin_date=self.start_date, end_date=self.end_date)

        self.data_handler_copy = self.data_handler.copy()
        self.feature_handler = feature_handler if feature_handler else FeatureExtractor(self.data_handler_copy)
        self.signal_processors = signal_processors

        self.initial_capital = initial_capital
        self.total_equity = initial_capital
        self.balances_usdt = initial_capital
        self.balances_symbol = {symbol: 0.0 for symbol in self.symbols}
        self.quantity_symbols = {symbol: 0.0 for symbol in self.symbols}

        #### portfolio manager, risk manager, order manager, strategy,
        #### they need to be initialized after the datahandler and feature handlers

        self.portfolio_manager = portfolio_manager
        self.assigned_percentage = {}
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.strategy = strategy

        self.equity_history = []
        self.balance_history = {symbol: [] for symbol in self.symbols}
        self.trade_logs = {symbol: [] for symbol in self.symbols}
        self.entry_prices = {symbol: -1 for symbol in self.symbols}

    def equity_balance(self, prices = None, trade = False):
        """
        Calculate and update equity and balances dynamically based on quantities and prices.
        """
        if not prices:
            prices={symbol: 0 for symbol in self.symbols}
        total_equity = self.balances_usdt  # Start with the USDT balance
        for symbol in self.symbols:
            price = prices[symbol]
            self.balances_symbol[symbol] = self.quantity_symbols[symbol] * price
            total_equity += self.balances_symbol[symbol]
        self.total_equity = total_equity
        if self.risk_manager: self.risk_manager.update_equity_balance(self.total_equity, self.balances_symbol, trade)
        if self.strategy: self.strategy.update_equity_balance(self.total_equity, self.balances_symbol, trade)
        return total_equity
    
    def update_assigned_percentage(self):
        self.assigned_percentage = self.portfolio_manager.get_assigned_percentage()
        self.risk_manager.update_assigned_percentage(self.assigned_percentage)
        self.strategy.update_assigned_percentage(self.assigned_percentage)

    def run_initialization(self):
        """
        Initialize the backtester by setting up equity, balances, and tools.
        """
        self.read_quantities()
        self.equity_balance()
        self.initialize_PortfolioManager()
        self.assigned_percentage = self.portfolio_manager.assigned_percentage
        self.initialize_Strategy(self.balances_symbol, self.data_handler_copy)
        self.set_entry_prices()

    def read_quantities(self):
        # read the existing qunatities in the account
        # Not needed for backtest
        pass

    def initialize_PortfolioManager(self):
        """
        Initialize the portfolio manager.
        """
        self.portfolio_manager = PortfolioManager(self.total_equity, self.balances_symbol, self.total_equity ,self.symbols, self.data_handler_copy)
        self.assigned_percentage = self.portfolio_manager.assigned_percentage


    def initialize_Strategy(self, quantities, data_handler):
        """
        Initialize the multi-asset strategy.
        """
        s_config = json.load(open('backtest/config/strategy.json', 'r'))
        m_config, r_config, d_config = {}, {}, {}

        for symbol in self.symbols:
            m_config[symbol] = s_config[symbol]['model']
            r_config[symbol] = s_config[symbol]['risk_manager']
            d_config[symbol] = s_config[symbol]['decision_maker']

        self.risk_manager = RiskManager(self.total_equity, self.balances_symbol, self.total_equity, self.assigned_percentage, r_config, self.data_handler_copy, self.signal_processors, self.feature_handler)
        self.risk_manager.initialize_singles()
        self.risk_manager.calculate_position()

        self.strategy = MultiAssetStrategy(self.total_equity, self.balances_symbol, self.total_equity, self.assigned_percentage, data_handler, self.risk_manager, m_config, d_config,
                                        self.feature_handler, self.signal_processors)
        self.strategy.initialize_singles()

    def set_entry_prices(self):
        self.cal_entry_prices()
        for symbol in self.symbols:
            self.entry_prices[symbol] = self.entry_prices[symbol]
        self.risk_manager.set_entry_price(self.entry_prices)
        
    def cal_entry_prices(self):
        """
        Calculates and sets the weighted average entry prices for all symbols.
        Updates self.entry_price as a dictionary {symbol: price}.
        """
        if not self.trade_logs:
            pass

        for symbol, trades in self.trade_logs.items():
            total_cost = 0.0
            total_quantity = 0.0

            # Process the trade log for the symbol anti-chronologically
            for trade in reversed(trades):
                trade_qty = trade["quantity"]
                trade_price = trade["price"]
                trade_type = trade["order"]  # 'buy' or 'sell'

                if trade_type == "buy":
                    total_cost += trade_qty * trade_price
                    total_quantity += trade_qty
                elif trade_type == "sell":
                    total_quantity -= trade_qty
                    # If position is fully closed, terminate early
                    if total_quantity <= 0:
                        total_quantity = 0
                        total_cost = 0
                        break

            self.entry_prices[symbol] = total_cost / total_quantity if total_quantity > 0 else 0.0


    def run_backtest(self):
        """
        Runs the backtest for all symbols over the specified date range.
        """
        print(f"Initializing multi-asset backtest from {self.start_date} to {self.end_date}...")

        # Load and prepare data
        self.data_handler.load_data(interval_str=self.interval_str, begin_date=self.start_date, end_date=self.end_date)
        # Initialize equity and balances
        """
        For quick testing, we limit the data to the last 3000 rows.
        for symbol in self.symbols:
            self.data_handler.cleaned_data[symbol] = self.data_handler.cleaned_data[symbol].tail(3000)
        """
        self.equity_balance()
        self.equity_history.append(self.total_equity)
        for symbol in self.symbols:
            self.balance_history[symbol].append(self.balances_symbol[symbol])
        # Prepare indicators
        self.feature_handler.pre_run_indicators()
        i = 100
        backtest_len = len(self.data_handler.cleaned_data[self.symbols[0]])-i
        
        start_date = self.data_handler.cleaned_data[self.symbols[0]].index[0]
        end_date = self.data_handler.cleaned_data[self.symbols[0]].index[-1]
        print(f"Starting backtest from {start_date} to {end_date}.")

        start_index = max(0, i - self.window_size)
        self.current_date = self.data_handler.cleaned_data[self.symbols[0]].index[i]
        for symbol in self.symbols:
            self.data_handler_copy.cleaned_data[symbol] = self.data_handler.get_data_range(
                symbol, start_index=start_index, end_index=i
            )
        self.feature_handler.pre_run_indicators()

        # Iterate through the data for each symbol
        for i in tqdm(range(self.window_size, len(self.data_handler.cleaned_data[self.symbols[0]]) - 1)):
            self.current_date = self.data_handler.get_data(self.symbols[0]).index[i]
            latest_data = {}
            latest_prices = {}
            for symbol in self.symbols:
                # Update data for the current window
                self.data_handler_copy.cleaned_data[symbol] = self.data_handler.get_data_range(
                    symbol, start_index=i - self.window_size, end_index=i
                )
                latest_data[symbol] = self.data_handler_copy.cleaned_data[symbol].iloc[-1]
                latest_prices[symbol] = latest_data[symbol]['close']
            self.equity_balance(latest_prices)
            self.data_handler_copy.notify_subscribers(latest_data)
            self.update_assigned_percentage()

            market_order = self.strategy.run_strategy_market()
            for symbol in self.symbols:
                market_order[symbol]['price'] = latest_data[symbol]['close']
                if market_order[symbol]['signal'] == 'hold':
                    self.balance_history[symbol].append(self.balances_symbol[symbol])
                    continue
            # Execute orders for the symbol
                trade_log = self.execute_order(symbol, market_order[symbol])
                self.trade_logs[symbol].append(trade_log)
                self.balance_history[symbol].append(self.balances_symbol[symbol])
                self.equity_balance(latest_prices, trade=True)

            self.equity_history.append(self.total_equity)
            
            # self.log_state(self.current_date)

        print("Multi-asset backtest completed. Evaluating performance...")

        # Evaluate performance metrics
        self.performance_metrics = self.evaluate_performance()

        # Save performance metrics
        self.save_metrics()

        print("Backtest results saved.")


    def execute_order(self, symbol, order):
        """
        Execute buy/sell orders for a given symbol.
        """
        order_type = order['signal']
        amount = order['amount']
        price = order['price']

        if order_type == "buy":
            cost = amount * price
            if self.balances_usdt < cost/(1+fee):
                amount = self.balances_usdt/price/(1+fee)
                cost = self.balances_usdt
            self.balances_usdt -= cost
            self.quantity_symbols[symbol] += amount
            self.balances_symbol[symbol] = self.quantity_symbols[symbol] * price
        elif order_type == "sell":
            if self.quantity_symbols[symbol] < amount:
                amount = self.quantity_symbols[symbol]
            self.balances_usdt += amount * price * (1-fee)
            self.quantity_symbols[symbol] -= amount
            self.balances_symbol[symbol] = self.quantity_symbols[symbol] * price
        
        return {
            "symbol": symbol,
            "order": order_type,
            "quantity": amount,
            "price": price,
            "date": self.data_handler.get_last_data(symbol).name,
            "balance": self.balances_symbol[symbol]}

    def cal_entry_prices(self):
        """
        Calculates and sets the weighted average entry prices for all symbols.
        Uses self.quantity_symbols to determine current asset holdings.
        
        Updates self.entry_prices as a dictionary {symbol: price}.
        If a symbol has no position, assigns -1.
        If a symbol's trade log is missing or its position is fully closed, assigns 0.0.
        """
        self.entry_prices = {}  # Ensure dictionary is initialized

        # Check if quantity tracking is available
        if not hasattr(self, 'quantity_symbols') or not self.quantity_symbols:
            print("Warning: No asset quantities found. Entry prices cannot be calculated.")
            return  

        # Process each symbol separately
        for symbol in self.quantity_symbols.keys():
            total_cost = 0.0
            total_quantity = self.quantity_symbols.get(symbol, 0.0)  # Get the current asset quantity
            threshold = 1e-6  # Small value to handle floating-point precision errors

            # If the asset quantity is None or too small, set entry price to -1 (no position)
            if total_quantity is None or total_quantity < threshold:
                print(f"Warning: No position for {symbol}. Entry price set to -1.")
                self.entry_prices[symbol] = -1
                continue  # Skip further processing for this symbol

            # If there is no trade history for this symbol, set to default 0.0
            if symbol not in self.trade_logs or not self.trade_logs[symbol]:
                print(f"Info: No trade history available for {symbol}. Using default entry price.")
                self.entry_prices[symbol] = 0.0
                continue  

            # Process the trade log for the symbol anti-chronologically (most recent first)
            for trade in reversed(self.trade_logs[symbol]):
                trade_qty = trade["amount"]  # Ensure consistency with original single-symbol version
                trade_price = trade["price"]
                trade_type = trade["type"]  # 'buy' or 'sell'

                if trade_type == "buy":
                    total_cost += trade_qty * trade_price
                    total_quantity += trade_qty  # Accumulate bought quantity
                elif trade_type == "sell":
                    total_quantity -= trade_qty  # Reduce available quantity

                    # If the position is fully closed, terminate early
                    if total_quantity <= 0:
                        total_quantity = 0
                        total_cost = 0
                        break

            # Calculate and store the weighted average entry price
            if total_quantity > 0:
                self.entry_prices[symbol] = total_cost / total_quantity
            else:
                self.entry_prices[symbol] = 0.0  # Position fully closed


    def log_state(self, current_date):
        # for future development usage
        """
        Log the current state of equity, balances, and quantities.
        """
        self.equity_history.append({
            "date": current_date,
            "total_equity": self.total_equity
        })
        for symbol in self.symbols:
            self.balance_history[symbol].append({
                "date": current_date,
                "balance": self.balances_symbol[symbol],
                "quantity": self.quantity_symbols[symbol]
            })

    def evaluate_performance(self):
        """
        Evaluate the performance of the backtest using the MultiSymbolStrategyEvaluator.
        """
        # Create an evaluator instance with backtest data
        evaluator = MultiSymbolStrategyEvaluator(
            trade_logs=self.trade_logs,
            equity_history=self.equity_history,
            asset_balance_history=self.balance_history,
            initial_balance=self.initial_capital,
            interval_str=self.interval_str
        )

        # Calculate comprehensive performance metrics
        return evaluator.calculate_metrics()

    def save_metrics(self, output_path="backtest/performance/multi_symbol/"):
        """
        Save performance metrics to JSON files.
        """
        os.makedirs(output_path, exist_ok=True)

        # Evaluate overall performance
        performance = self.evaluate_performance()

        # Save overall performance metrics
        overall_path = os.path.join(output_path, "overall_performance.json")
        with open(overall_path, "w") as overall_file:
            json.dump(performance, overall_file, indent=4)
        print(f"Saved overall performance metrics to {overall_path}.")

        # Save individual symbol performance metrics
        for symbol, metrics in performance['Symbol ROI (%)'].items():
            file_name = f"{symbol}_performance.json"
            file_path = os.path.join(output_path, file_name)
            with open(file_path, "w") as file:
                json.dump({"ROI (%)": metrics}, file, indent=4)
            print(f"Saved performance metrics for {symbol} to {file_path}.")

    


if __name__ == "__main__":
    # Example of initializing and running the Backtester
    strategy = ...  # Replace with your custom strategy
    data_handler = HistoricalDataHandler()
    portfolio_manager = PortfolioManager()
    risk_manager = RiskManager()


    backtester = MultiAssetBacktester(
        Strategy=strategy,
        data_handler=data_handler,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        initial_capital=100000.0
    )
    backtester.run_backtest("2022-01-01", "2023-01-01")
    performance = backtester.evaluate_performance()
    print("Performance Metrics:", performance)
    backtester.save_results()