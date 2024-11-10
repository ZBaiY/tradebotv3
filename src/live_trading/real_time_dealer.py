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
import src.strategy as strategy
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
    def __init__(self, strategy, capital_allocator, risk_manager, portfolio_manager, api_path = 'config/api_config.json', log_dir='/trade/logs', log_file='real_time_dealer.log'):
        self.data_handler = RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')  
        
        self.api = json.load(open(api_path, 'r'))
        api_key = self.api.get('api_key', None)
        api_secret = self.api.get('api_secret', None)
        self.client = Client(api_key, api_secret)
        # Initialize OrderManager with API credentials and logging configurations
        self.OrderManager = OrderManager(self.client)
        self.Strategy = strategy  
        self.CapitalAllocator = capital_allocator
        self.RiskManager = risk_manager
        self.PortfolioManager = portfolio_manager
    

        # Initialize logging handler for the dealer
        self.logger = LoggingHandler(log_dir=log_dir, log_file=log_file).logger
        self.logger.info("RealtimeDealer initialized.")

        self.symbols = self.data_handler.symbols
        self.equity = None
        self.balances_symbol = None
        self.balances_str = None
        self.balances_symbol_fr = None # can be touched in total
        self.entry_prices = {}
        self.equity_balance()
        self.equity_balance_tools()
        self.set_symbols(self.symbols)
        

        self.is_running = False 
    def set_symbols(self, symbols):
        self.Strategy.set_symbols(symbols)
        self.RiskManager.set_symbols(symbols)
        self.PortfolioManager.set_symbols(symbols)
        self.CapitalAllocator.set_symbols(symbols)
    
    
    
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
                self.balances_symbol_fr[asset] = free

        return total_equity
    
    def set_entry_price(self):
        for symbol in self.symbols:
            self.entry_prices[symbol] = self.entry_price(symbol)
    
    def entry_price(self, symbol):
        """
        Calculate the weighted average entry price for a current open position of a specific symbol.
        :param symbol: The symbol to calculate the entry price for (e.g., 'BTCUSDT').
        :return: The average entry price or None if no open position.
        """
        past_trades = self.OrderManager.fetch_past_trades_from_api()
        total_bought_qty = 0.0
        total_cost = 0.0
        remaining_qty = 0.0

        for order_id, order in past_trades.items():
            # Only consider trades for the given symbol that are fully filled
            if order['symbol'] == symbol and order['status'] == "FILLED":
                trade_qty = float(order['executedQty'])
                trade_price = float(order['price'])
                trade_side = order['side']  # 'BUY' or 'SELL'

                if trade_side == 'BUY':
                    total_cost += trade_qty * trade_price  # Accumulate the cost
                    total_bought_qty += trade_qty  # Accumulate the quantity bought
                    remaining_qty += trade_qty  # Add to the open position
                elif trade_side == 'SELL':
                    remaining_qty -= trade_qty  # Reduce the open position by the sold quantity
                    # If more quantity is sold than was bought, reset total cost and quantities
                    if remaining_qty < 0:
                        remaining_qty = 0
                        total_cost = 0
                        total_bought_qty = 0

        # Calculate weighted average entry price based on remaining open position
        if remaining_qty > 0:
            weighted_entry_price = total_cost / total_bought_qty
            self.logger.info(f"Calculated entry price for {symbol}: {weighted_entry_price}")
            return weighted_entry_price
        else:
            self.logger.info(f"No open position found for {symbol}. Entry price calculation not applicable.")
            return 0.0
        
    def equity_balance_tools(self):
        self.RiskManager.set_equity(self.equity)
        self.RiskManager.set_balances(self.balances_symbol_fr)   
        self.Strategy.set_equity(self.equity)
        self.Strategy.set_balances(self.balances_symbol_fr)
        self.PortfolioManager.set_equity(self.equity)
        self.PortfolioManager.set_balances(self.balances_symbol_fr)
        self.CapitalAllocator.set_equity(self.equity)
        self.CapitalAllocator.set_balances(self.balances_symbol_fr)

    def set_entry_price(self):
        self.RiskManager.set_entry_price(self.entry_prices)

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

        while self.is_running:
            self.data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
            self.data_handler.notify_subscribers()
            # Includes running the data processing (for the processor who need it), feature extraction
            limit_signals, market_orders = self.strategy.run_strategy(self.data_handler)
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

            self.equity_balance()
            self.equity_balance_tools()
            self.set_entry_price()

            self.CapitalAllocator.action() # CapitalAllocator will adjust the equity and balance
            self.PortfolioManager.action() # PortfolioManager will adjust the equity between assets

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