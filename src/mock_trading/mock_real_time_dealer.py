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
from src.strategy.multi_asset_strategy import MultiAssetStrategy
# from src.live_trading.execution_handler import ExecutionHandler
from mock_trading.mock_order_manager import MockOrderManager


class MockRealtimeDealer:
    def __init__(self, datahandler=None, feature_module=None, signal_processors=None, log_dir='../../mock/logs', log_file='mock_real_time_dealer.log'):
        # Initialize mock data handlers
        self.data_handler = datahandler or RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')  
        self.features = feature_module or FeatureExtractor(self.data_handler)
        self.signal_processors = signal_processors
        if signal_processors is None and os.path.exists('config/processors.json'):
            processors_config = json.load(open('config/processors.json', 'r'))
            self.signal_processors = {}
            for processor_name, config in processors_config:
                column = config.get('column', None)
                self.signal_processors[processor_name] = SignalProcessor(self.data_handler, column=column)
                self.signal_processors[processor_name].initialize_processors(config)
        
        # Initialize logging
        self.logger = LoggingHandler(log_dir=log_dir, log_file=log_file).logger
        self.logger.info("RealtimeDealer initialized.")
        # Mock objects and variables
        self.OrderManager = MockOrderManager()
        self.equity = None  
        self.symbols = self.data_handler.symbols
        self.equity = None # total equity in USDT
        self.quantities = {} # total quantity in each symbol
        self.balances_symbol = {} # total balance in each symbol
        ###### This is the balances in USDT
        self.balances_str = {} # total balance in each symbol in string
        ####### the balances are in assets quantities not in USDT
        self.balances_symbol_fr = {} # free balance in each symbol
        ####### Free balances in USDT
        ####### Question, which one should be related to the risk manager?
        ####### For deciding postion is the free one, for portfolio management, the total one.
        self.allocation_cryp = None ## Total allocation to crypto
        self.assigned_percentage = {}
        self.entry_prices = {}


        # Initialize portfolio and strategy
        self.CapitalAllocator = None
        self.PortfolioManager = None
        self.RiskManager = None
        self.Strategy = None

        self.freezing_times = 0 ### used for checking the whether the trade is frozen or not

        # Preparing to run the system
        self.is_running = False 


########################### Block 1: equity, balances, entry prices, asigned capitals ########################################
    def equity_balance(self, trade=False):
        
        info = self.OrderManager.get_account_info()
        self.balances_str = info['balances']
        self.equity = self.calculate_equity(self.balances_str)
        if self.RiskManager: self.RiskManager.update_equity_balance(self.equity, self.balances_symbol_fr, trade)
        if self.Strategy: self.Strategy.update_equity_balance(self.equity, self.balances_symbol_fr, trade)
        if self.PortfolioManager: self.PortfolioManager.update_equity_balance(self.equity, self.balances_symbol)
        if self.CapitalAllocator: self.CapitalAllocator.set_equity(self.equity)

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
                self.quantities[asset] = total
            else:
                symbol = f"{asset}USDT"
                price = self.get_asset_price(asset, 'USDT')
                total_equity += total * price
                self.quantities[asset] = total
                self.balances_symbol[symbol] = total * price
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
        past_trades = self.OrderManager.fetch_past_trades_from_api(symbol)
        
        # Reverse order to process trades anti-chronologically (most recent first)
        past_trades = list(reversed(past_trades))
        # Get the current balance of the asset (BTC)
        asset = symbol.replace('USDT', '')
        remaining_qty = self.quantities[asset]  # Total BTC we currently hold
        original_qty = remaining_qty  # Keep track of original balance
        total_cost = 0.0
        total_bought_qty = 0.0
        threshold = 1e-5  # Small threshold for rounding errors

        if remaining_qty is None or remaining_qty < threshold:
            self.logger.info(f"No open position for {symbol}. Returning -1.")
            return -1
        

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

    
    def update_assigned_percentage(self):
        self.assigned_percentage = self.PortfolioManager.get_assigned_percentage()
        self.RiskManager.update_assigned_percentage(self.assigned_percentage)
        self.Strategy.update_assigned_percentage(self.assigned_percentage)


    def equity_balance_tools(self, trade=False):
        self.RiskManager.set_equity(self.equity)
        self.RiskManager.set_balances(self.balances_symbol_fr, trade)   
        self.Strategy.set_equity(self.equity) # Maybe redundant
        self.Strategy.set_balances(self.balances_symbol_fr, trade)  # Maybe redundant
        self.PortfolioManager.set_equity(self.equity)
        self.PortfolioManager.set_balances(self.balances_symbol)
        self.CapitalAllocator.set_equity(self.equity)
        self.CapitalAllocator.set_balances(self.balances_symbol)
    

########################### Block 1: equity, balances, entry prices, asigned capitals ########################################

########################### Block 2: Initialization for the tools ########################################
    def reset_config(self):
        self.OrderManager.reset_config()

    def run_initialization(self):
        """
        Fist read the current equity and balance, and portfolio manager from logs
        Then set the equity and balance to the strategy, risk_manager, and portfolio
        Then Initialize the strategy, risk_manager singles
        Then set the entry prices

        After this function, all the tools are ready to run
        """
        self.equity_balance()
        self.initialize_CapitalAllocator(self.equity)
        self.initialize_PortfolioManager() 
        self.initialize_Straegy(self.equity, self.balances_symbol_fr, self.allocation_cryp, self.assigned_percentage)          # Model, RiskManager is initialized here
        self.set_assigned_percentage(self.assigned_percentage) # percentage here is smaller than the 1.0
        self.set_entry_prices()
        
    
    def initialize_CapitalAllocator(self, equity):
        self.CapitalAllocator = CapitalAllocator(equity)
        self.allocation_cryp = self.CapitalAllocator.get_allocation_cryp() 

    def initialize_PortfolioManager(self):
        """
        Initialize the portfolio manager.
        """
        self.PortfolioManager = PortfolioManager(self.equity, self.balances_symbol, self.allocation_cryp ,self.symbols, self.data_handler)
        self.assigned_percentage = self.PortfolioManager.assigned_percentage

    
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
        self.RiskManager = RiskManager(equity, balances, allocation_cryp, assigned_percentage, r_config, self.data_handler, self.signal_processors, self.features)
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
        
    def simulate_equity_balance(self):
        """
        Simulates equity and balance updates.
        """
        self.equity = 100000  # Simulated fixed equity
        self.balances_symbol_fr = {symbol: self.equity / len(self.symbols) for symbol in self.symbols}
        self.logger.info(f"Simulated equity balance: {self.equity}, Balances: {self.balances_symbol_fr}")

    def simulate_entry_prices(self):
        """
        Simulates the setting of entry prices.
        """
        for symbol in self.symbols:
            self.entry_prices[symbol] = self.data_handler.get_last_data(symbol)['close']
        self.RiskManager.set_entry_price(self.entry_prices)
        self.logger.info(f"Simulated entry prices: {self.entry_prices}")


    def start(self):

        self.is_running = True
        next_fetch_time, last_fetch_time = self.data_handler.pre_run_data()
        self.logger.info("Starting RealtimeDealer.")
        self.run_initialization()
        self.features.pre_run_indicators()
        # print("now: ", datetime.now(timezone.utc))
        # print("next_fetch_time: ", next_fetch_time)
        # print("last_fetch_time: ", last_fetch_time)
        
        while self.is_running:
            new_data = self.data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
            # print("new data: ", new_data)
            last_fetch_time = next_fetch_time
            self.equity_balance()
            self.data_handler.notify_subscribers(new_data)
            # self.update_assigned_percentage()  one needs to do rebalance, so only do this per month.
            self.set_entry_prices() 
            market_orders = self.Strategy.run_strategy_market()
            # print(market_orders)
            # input ("mock_real 279, Press Enter to continue...")
            # Includes running data processing, feature extraction, model prediction, generating signals, 
            # and applying stop loss/take profit to determine amount to buy/sell.
            """
            Example return values from the strategy:
            limit_signals = {'BTCUSDT': {'signal': 'buy', 'amount': 0.1, 'price': 10000.0}, 'ETHUSDT': {'signal': 'sell', 'amount': 0.2, 'price': 500.0}}
            market_orders = {'BTCUSDT': {'signal': 'buy', 'amount': 0.1}, 'ETHUSDT': {'signal': 'sell', 'amount': 0.2}}
            """

            # Handle limit signals
            """
            for symbol, signal in limit_signals.items():
                if signal['signal'] in ['buy', 'sell']:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type=signal['signal'],
                        amount=signal['amount'],
                        price=signal['price']
                    )
            """
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
                    self.equity_balance(trade=True)
            # Execute the predicted signals
            print("current_price: ", {'ETHUSDT': self.data_handler.get_last_data('ETHUSDT')['close'], 'BTCUSDT':self.data_handler.get_last_data('BTCUSDT')['close']})
            print("stop_loss: ", stop_loss)
            print("take_profit: ", take_profit)
            # input ("mock_real 319, Press Enter to continue...")
            """for symbol in self.symbols:
                free_balance = self.balances_symbol_fr[symbol]
                current_price = self.data_handler.get_last_data(symbol)['close']
                if stop_loss[symbol] >= current_price:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type='sell',
                        amount=free_balance,
                        price=-1
                    )
                if take_profit[symbol] <= current_price:
                    self.OrderManager.create_order(
                        symbol=symbol,
                        order_type='sell',
                        amount=free_balance,
                        price=-1
                    )"""

            # Check for any stop-loss or take-profit conditions and execute orders
            # After executing the orders, update the equity and balances
            
            
            
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
            next_fetch_time = self.data_handler.calculate_next_grid(now)
            # self.monitor_system_health()

            sleep_duration = (next_fetch_time - now).total_seconds() + 1
            self.logger.info(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
            time.sleep(sleep_duration)



    def stop(self):
        self.is_running = False
        self.logger.info("Stopping MockRealtimeDealer.")

    def reset_trader(self):
        pass

if __name__ == "__main__":
    dealer = MockRealtimeDealer()
    try:
        dealer.start()
    except KeyboardInterrupt:
        dealer.stop()