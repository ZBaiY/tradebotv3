# real_time_dealer.py
import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import logging
from datetime import datetime, timezone, timedelta
import time
import requests
import json
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.data_handling.real_time_data_handler import RealTimeDataHandler, LoggingHandler
from src.portfolio_management.risk_manager import RiskManager
from src.portfolio_management.capital_allocator import CapitalAllocator
from src.portfolio_management.portfolio_manager import PortfolioManager
from src.feature_engineering.feature_extractor import FeatureExtractor
from src.signal_processing.signal_processor import SignalProcessor
import src.strategy.multi_asset_strategy as MultiAssetStrategy
# from src.live_trading.execution_handler import ExecutionHandler
from src.live_trading.order_manager import OrderManager

"""
We wrote something for limit orders 
but in the beginning of the project,
We don't have any limit orders. We will use market orders for now.
So the limit order logic is imcomplete.
Like dealing with the order status, canceling the old order, etc.
"""


"""
To initialize the risk_manager and strategy, we need to read the config files.
"""

class RealtimeDealer:
    def __init__(self, datahandler=None, feature_module=None, signal_processors=None, api_path = 'config/api_config.json', log_dir='/trade/logs', log_file='real_time_dealer.log'):
        
        # Initialize the data baths
        self.data_handler = datahandler
        if datahandler is None:
            self.data_handler = RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')  
        self.features = feature_module
        if feature_module is None:
            self.features = FeatureExtractor(self.data_handler)
        self.signal_processors = signal_processors
        if signal_processors is None:
            processors_config = json.load(open('config/processors.json', 'r'))
            self.signal_processors = {}
            for processor_name, config in processors_config:
                column = config.get('column', None)
                self.signal_processors[processor_name] = SignalProcessor(self.data_handler, column=column)
                self.signal_processors[processor_name].initialize_processors(config)
        
        # Initialize the Binance API client and OrderManager

        self.api = json.load(open(api_path, 'r'))
        api_key = self.api.get('api_key', None)
        api_secret = self.api.get('api_secret', None)
        self.client = Client(api_key, api_secret)
        self.OrderManager = OrderManager(self.client)

        # Initialize logging handler for the dealer
        self.logger = LoggingHandler(log_dir=log_dir, log_file=log_file).logger
        self.logger.info("RealtimeDealer initialized.")

        # Initialize the variables
        self.symbols = self.data_handler.symbols
        self.equity = None # total equity in USDT
        self.balances_symbol = None # total balance in each symbol
        self.balances_str = None # total balance in each symbol in string
        self.balances_symbol_fr = None # free balance in each symbol
        self.allocation_cryp = None ## Total allocation to crypto
        self.assigned_percentage = None
        self.entry_prices = {}

        # Initialize the tools
        self.CapitalAllocator = None
        self.PortfolioManager = None
        self.RiskManager = None
        self.Strategy = None
        
        self.freezing_times = 0 ### used for checking the whether the trade is frozen or not

        # Preparing to run the system
        self.is_running = False 

########################### Block 1: equity, balances, entry prices, asigned capitals ########################################
    def equity_balance(self):
        info = self.OrderManager.get_account_info()['balance']
        self.balances_str = info['balances']
        self.equity = self.calculate_equity(self.balances_str)

    # Assets e.g. BTC, are not symbols, symbols are trading pairs e.g. BTCUSDT
    # Assests are used to calculate the equity in Binance API
    def calculate_equity(self, balances):
        total_equity = 0.0
        for balance in balances:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked

            if asset == 'USDT':
                total_equity += total
            else:
                symbol = f"{asset}USDT"
                price = self.get_asset_price(asset, 'USDT')
                total_equity += total * price
                self.balances_symbol[symbol] = total
                self.balances_symbol_fr[symbol] = free * price

        return total_equity
    
    def set_entry_prices(self):
        entry_prices = {}
        for symbol in self.symbols:
            entry_prices[symbol] = self.entry_price(symbol)
        self.entry_prices = entry_prices
        self.RiskManager.set_entry_price(self.entry_prices)

    def entry_price(self, symbol):
        """
        Calculate the weighted average entry price for a current open position of a specific symbol.
        :param symbol: The symbol to calculate the entry price for (e.g., 'BTCUSDT').
        :return: The average entry price or current market price if insufficient trade history.
        """
        past_trades = self.OrderManager.fetch_past_trades_from_api()
        
        # Reverse order to process trades anti-chronologically (most recent first)
        past_trades = list(reversed(past_trades))

        # Get the current balance of the asset (BTC)
        remaining_qty = self.balances_symbol[symbol]  # Total BTC we currently hold
        if remaining_qty is None or remaining_qty < threshold:
            self.logger.info(f"No open position for {symbol}. Returning -1.")
            return -1
        original_qty = remaining_qty  # Keep track of original balance
        total_cost = 0.0
        total_bought_qty = 0.0
        threshold = 1e-5  # Small threshold for rounding errors

        for order in past_trades:
            if order['symbol'] == symbol:
                trade_qty = float(order['executedQty'])
                trade_price = float(order['price'])
                trade_side = order['side']  # 'BUY' or 'SELL'

                if trade_side == 'buy':
                    total_cost += trade_qty * trade_price  # Accumulate cost
                    total_bought_qty += trade_qty  # Accumulate bought quantity
                    remaining_qty -= trade_qty  # Reduce the amount still needing reconciliation
                elif trade_side == 'sell':
                    remaining_qty += trade_qty  # Selling increases the balance we need to account for

                # Stop early when the accounted amount matches the balance within a small margin
                if remaining_qty < -1 * threshold:
                    self.logger.warning(f"Trade log inconsistency detected for {symbol}. "
                                        f"Remaining quantity dropped below zero. Stopping calculation.")
                    break
                if remaining_qty < threshold:
                    break
        # Calculate weighted average entry price
        if total_bought_qty > threshold:
            weighted_entry_price = total_cost / total_bought_qty
            self.logger.info(f"Calculated entry price for {symbol}: {weighted_entry_price}")
            return weighted_entry_price
        else:
            # Fallback: Use current market price if no valid entry price is found
            market_price = self.data_handler.get_last_data(symbol)['close']
            self.logger.info(f"Using market price as fallback for {symbol}: {market_price}")
            return market_price
       

    def set_assigned_percentage(self, assigned_percentage):
        self.assigned_percentage = assigned_percentage
        self.Strategy.set_assigned_percentage(assigned_percentage)
        self.RiskManager.set_assigned_percentage(assigned_percentage)
    
    def update_assigned_percentage(self, assigned_percentage):
        self.assigned_percentage = assigned_percentage
        self.RiskManager.set_assigned_percentage(assigned_percentage)
        self.Strategy.set_assigned_percentage(assigned_percentage)

    


    def equity_balance_tools(self):
        self.RiskManager.set_equity(self.equity)
        self.RiskManager.set_balances(self.balances_symbol_fr)   
        self.Strategy.set_equity(self.equity) # Maybe redundant
        self.Strategy.set_balances(self.balances_symbol_fr)  # Maybe redundant
        self.PortfolioManager.set_equity(self.equity)
        self.PortfolioManager.set_balances(self.balances_symbol_fr)
#        self.CapitalAllocator.set_equity(self.equity)
#        self.CapitalAllocator.set_balances(self.balances_symbol_fr)

    
    def update_equity_balances(self):
        self.RiskManager.update_equity(self.equity)
        self.Strategy.update_equity(self.equity) # Maybe redundant
        self.RiskManager.update_balances(self.balances_symbol_fr)
        self.Strategy.update_balances(self.balances_symbol_fr) # Maybe redundant

########################### Block 1: equity, balances, entry prices, asigned capitals ########################################

########################### Block 2: Initialization for the tools ########################################
    def run_initialization(self):
        """
        Fist read the current equity and balance, and portfolio manager from logs
        Then set the equity and balance to the strategy, risk_manager, and portfolio
        Then Initialize the strategy, risk_manager singles
        Then set the entry prices

        After this function, all the tools are ready to run
        """
        self.equity_balance()
        self.initialize_CapitalAllocator(self.equity, self.balances_symbol_fr)
        self.allocation_cryp = self.CapitalAllocator.get_allocation_cryp() 
        self.initialize_PortfolioManager(self.equity, self.balances_symbol_fr, self.allocation_cryp, self.symbols) 
        self.assigned_percentage = self.PortfolioManager.get_assigned_percentage()
        self.initialize_Straegy(self.equity, self.balances_symbol_fr, self.allocation_cryp, self.assigned_percentage)          # Model, RiskManager is initialized here
        self.set_assigned_percentage(self.assigned_percentage) # percentage here is smaller than the 1.0
        self.set_entry_prices()
        
    def initialize_CapitalAllocator(self):
        self.CapitalAllocator = CapitalAllocator()
    def initialize_PortfolioManager(self, equity, balances_symbol_fr, allocation_cryp, symbols):
        self.PortfolioManager = PortfolioManager(equity, balances_symbol_fr, allocation_cryp, symbols)
        

    def initialize_Straegy(self, equity, balances, allocation_cryp, assigned_percentage):
        # Read from json file
        s_config = json.load(open('config/strategy.json', 'r'))
        m_config = {}
        r_config = {}
        d_config = {}
        for symbol in self.symbols:
            m_config[symbol] = s_config[symbol]['model']
            r_config[symbol] = s_config[symbol]['risk_manager']
            d_config[symbol] = s_config[symbol]['decision_maker']

        #### will set the equity, balances, assigned_capitals, initialize the singles
        self.RiskManager = RiskManager(equity, balances, allocation_cryp, assigned_percentage, s_config, self.data_handler, self.signal_processors, self.features)
        self.RiskManager.initialize_singles()
        self.RiskManager.calculate_position()
        #### will set the equity, balances, assigned_capitals, initialize the models
        self.Strategy = MultiAssetStrategy(equity, balances, allocation_cryp, assigned_percentage, self.data_handler, self.RiskManager, m_config, d_config, self.features, self.signal_processors)
        self.Strategy.initialize_singles()

########################### Block 2: Initialization for the tools ########################################


    def get_asset_price(self, asset, quote):
        try:
            response = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={asset}{quote}')
            data = response.json()
            return float(data['price'])
        except Exception as e:
            self.logger.error(f"Error retrieving price for {asset}{quote}: {e}")
            return 0.0

    def start(self):

        self.is_running = True
        next_fetch_time, last_fetch_time = self.data_handler.pre_run_data()
        self.logger.info("Starting RealtimeDealer.")
        self.run_initialization()

        while self.is_running:
            self.data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
            self.data_handler.notify_subscribers()
            # Includes running the data processing (for the processor who need it), feature extraction
            market_orders = self.Strategy.run_strategy_market()
            # Includes running the model prediction, generating signals, 
            # and applying stop loss/take profit to determine amount to buy/sell
            """
            Example return values from the strategy:
            limit_signals = {
                'BTCUSDT': {'signal': 'buy', 'amount': 0.1, 'price': 10000.0},
                'ETHUSDT': {'signal': 'sell', 'amount': 0.2, 'price': 500.0}
            }
            market_orders = {
                'BTCUSDT': {'signal': 'buy', 'amount': 0.1},
                'ETHUSDT': {'signal': 'sell', 'amount': 0.2}
            }
            """

            # Handle limit signals
            """for symbol, signal in limit_signals.items():
                if signal['signal'] in ['buy', 'sell']:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type=signal['signal'],
                        amount=signal['amount'],
                        price=signal['price']
                    )"""
            stop_loss = self.RiskManager.get_stop_loss()
            take_profit = self.RiskManager.get_take_profit()
            # Handle market orders
            for symbol, signal in market_orders.items():
                if signal['signal'] in ['buy', 'sell']:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type=signal['signal'],
                        amount=signal['amount'],
                        price=-1  # For market order, use current price
                    )
            # Execute the predicted signals

            for symbol in self.symbols:
                free_balance = self.balances_symbol_fr[symbol]
                current_price = self.data_handler.get_last_data(symbol)['close']
                if stop_loss >= current_price:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type='sell',
                        amount=free_balance,
                        price=-1
                    )
                if take_profit <= current_price:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type='sell',
                        amount=free_balance,
                        price=-1
                    )
            # Check for any stop-loss or take-profit conditions and execute orders
            # After executing the orders, update the equity and balances
            self.equity_balance()
            self.update_equity_balances()
            self.set_entry_prices()
            
            ######
            # Missing functions: check the market orders completion
            # check the market opening
            # check new months --- organizing the data, and rebalancing
            # check the system health
            # check new weeks --- for updating the scalers
            ######

            """
            check months, if new month
            self.CapitalAllocator.action() # CapitalAllocator will adjust the equity and balance
            self.PortfolioManager.action() # PortfolioManager will adjust the equity between assets
            new_equity, new_assigned_capitals =self.rebalance_portfolio()  
            # self.rebalance_portfolio() will not only will rebalance, if order needs to be placed, it will also place the order

            self.update_equity_balances()
            self.update_assigned_capitals(new_assigned_capitals)
            """
            
            now = datetime.now(timezone.utc)
            next_fetch_time = self.calculate_next_grid(now)
            self.monitor_system_health()

            sleep_duration = (next_fetch_time - now).total_seconds() + 1
            self.logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
            time.sleep(sleep_duration)



    def calculate_next_grid(self, current_time):
        next_time = current_time + timedelta(minutes=1)
        self.logger.info(f"Next fetch time calculated: {next_time}")
        return next_time

    def monitor_system_health(self):
        if not self.data_handler.is_healthy():
            self.logger.warning("Data Handler issue detected. Restarting system.")
            self.restart_system()
        else:
            self.logger.info("System health check passed.")

    def restart_system(self):
        self.logger.info("Restarting system due to detected issue.")
        self.stop()
        self.start()

    def stop(self):
        self.is_running = False
        self.logger.info("Stopping RealtimeDealer.")

    def get_entry_price(self, symbol):
        return self.entry_prices[symbol]
    

    def check_rebalance(self):
        """
        Check if the portfolio needs rebalancing based on the current asset allocation.
        """
        pass

    def rebalance_portfolio(self):
        """
        This function will be called periodically to rebalance the portfolio based on the current asset allocation.
        Rebalance the portfolio based on the current asset allocation.
        """
        """Holder for the rebalance logic."""
        new_allocations = ...
        self.Strategy.set_assigned_percentage(new_allocations)
        self.RiskManager.set_assigned_percentage(new_allocations)
