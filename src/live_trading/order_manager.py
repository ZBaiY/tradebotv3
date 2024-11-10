# live_trading/execution_handler.py
import logging
import os
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException
import json
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handling.real_time_data_handler import LoggingHandler

class OrderManager:
    def __init__(self, client, log_dir='/trade/logs', log_file='order_manager.log', order_file='trade/orders.json'):
        # Initialize Binance client
        self.client = client
        self.active_orders = {}
        self.order_file = order_file
        # Initialize custom logging handler
        self.logger = LoggingHandler(log_dir=log_dir, log_file=log_file).logger
        self.logger.info("OrderManager initialized.")
        # Initialize the order file
        self.initialize_order_file()
    


    def initialize_order_file(self):
        """
        Initialize the order file by creating it if it doesn't exist.
        """
        try:
            if not os.path.exists(self.order_file):
                with open(self.order_file, 'w') as file:
                    json.dump({}, file)
                self.logger.info(f"Order file initialized: {self.order_file}")
        except Exception as e:
            self.logger.error(f"Error initializing order file: {e}")

    def create_order(self, symbol, order_type, amount, price=-1):
        """
        Place an order on Binance Futures.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param order_type: 'buy' or 'sell'
        :param amount: Amount to buy/sell
        :param price: Order price (-1 for market order)
        :return: Order dictionary if successful, None if failed
        """
        if order_type not in ['buy', 'sell']:
            self.logger.error("Invalid order type specified.")
            raise ValueError("Order type must be 'buy' or 'sell'.")

        # Determine the side based on order type
        side = Client.SIDE_BUY if order_type == 'buy' else Client.SIDE_SELL
        order_type = Client.ORDER_TYPE_MARKET if price == -1 else Client.ORDER_TYPE_LIMIT
        # Place the order on Binance Futures
        try:
            if order_type == Client.ORDER_TYPE_MARKET:
                # It will execute immediately
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=amount
                )
                self.logger.info(f"Created and executed {order_type} order: {order}")
            else:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=amount,
                    price=price,
                    timeInForce=Client.TIME_IN_FORCE_GTC
                )
                self.logger.info(f"Created {order_type} order: {order}")

            # Log and store the order
            
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error while creating order: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None
    
    def cancel_order(self, symbol, order_id):
        """
        Cancel a given order by order ID.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param order_id: The ID of the order to be canceled.
        :return: Cancellation result from Binance API
        """
        self.logger.info(f"Cancelling order for symbol={symbol} with ID: {order_id}")
        try:
            result = self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Order canceled: {result}")
            return result
        except BinanceAPIException as e:
            self.logger.error(f"Error canceling order: {e}")
            return None
    """Example canceled result:
    {
        "symbol": "BTCUSDT", "origClientOrderId": "myOrder1", "orderId": 123456789, "clientOrderId": "cancelMyOrder1",
        "price": "10000.00", "origQty": "1.00", "executedQty": "0.00", "cummulativeQuoteQty": "0.00",
        "status": "CANCELED", "timeInForce": "GTC", "type": "LIMIT", "side": "BUY"
    }
    """
    def check_order_status(self, symbol, order_id):
        """
        Check the status of a given order by order ID.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param order_id: The ID of the order to check.
        :return: Order status from Binance API
        """
        self.logger.info(f"Checking status for order ID: {order_id} on symbol={symbol}")
        try:
            status = self.client.futures_get_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Order status: {status}")
            return status
        except BinanceAPIException as e:
            self.logger.error(f"Error checking order status: {e}")
            return None
    """Example status:
    {
        "symbol": "BTCUSDT", "orderId": 123456789, "clientOrderId": "myOrder1", "price": "10000.00",
        "origQty": "1.00", "executedQty": "1.00", "status": "FILLED", "type": "LIMIT", "side": "BUY",
        "time": 1507725176595, "updateTime": 1507725176595, "isWorking": True
    }
    """

    def get_account_info(self):
        """
        Retrieve account information from the Binance API.
        :return: Account information from Binance API
        """
        self.logger.info("Retrieving account information")
        try:
            account_info = self.client.futures_account()
            self.logger.info(f"Account information: {account_info}")
            return account_info
        except BinanceAPIException as e:
            self.logger.error(f"Error retrieving account information: {e}")
            return None
    """Example account_info:
    {
        "makerCommission": 15, "takerCommission": 15, "buyerCommission": 0, "sellerCommission": 0,
        "canTrade": True, "canWithdraw": True, "canDeposit": True,
        "updateTime": 123456789, "accountType": "SPOT",
        "balances": [
            {"asset": "BTC", "free": "0.001", "locked": "0.000"},
            {"asset": "ETH", "free": "0.100", "locked": "0.000"}
        ],
        "permissions": ["SPOT"]
    }
    """

    def fetch_past_trades_from_api(self, symbol, limit=50):
        """
        Fetch past trades for a specific symbol from the Binance API.
        :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
        :param limit: Maximum number of trades to fetch (Binance allows up to 1000 at once)
        :return: List of trades or None if an error occurs
        """
        try:
            # For Binance Futures account trades
            trades = self.client.futures_account_trades(symbol=symbol, limit=limit)
            self.logger.info(f"Fetched {len(trades)} trades for {symbol} from Binance API.")
            return trades
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching trades from API for {symbol}: {e}")
            return None


    def add_order(self, order_id, order):
        """
        Add an active order to the tracking dictionary and write it to the file.
        :param order_id: Unique order ID
        :param order: Order dictionary
        """
        self.active_orders[order_id] = order
        self.logger.info(f"Order added to tracking: {order_id} -> {order}")
        self.write_order_to_file(order_id, order)

    def remove_order(self, order_id):
        """
        Remove an order from active tracking and update the file.
        :param order_id: The ID of the order to remove.
        """
        if order_id in self.active_orders:
            removed_order = self.active_orders.pop(order_id)
            self.logger.info(f"Order removed from tracking: {order_id} -> {removed_order}")
            self.update_order_file()
        else:
            self.logger.warning(f"Attempted to remove non-existent order ID: {order_id}")

    def update_order_file(self):
        """
        Update the order file with the current active orders.
        """
        try:
            with open(self.order_file, 'w') as file:
                json.dump(self.active_orders, file, indent=4)
            self.logger.info("Order file updated with current active orders.")
        except Exception as e:
            self.logger.error(f"Error updating order file: {e}")

    def write_order_to_file(self, order_id, order):
        """
        Write an order to the order file.
        :param order_id: Unique order ID
        :param order: Order dictionary
        """
        try:
            with open(self.order_file, 'r+') as file:
                orders = json.load(file)
                orders[order_id] = order
                file.seek(0)
                json.dump(orders, file, indent=4)
            self.logger.info(f"Order written to file: {order_id} -> {order}")
        except Exception as e:
            self.logger.error(f"Error writing order to file: {e}")

    