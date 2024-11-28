import logging
import os
import sys
import requests
from datetime import datetime
import time
import json
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handling.real_time_data_handler import LoggingHandler


class MockOrderManager:
    def __init__(self, mock_data_file='mock_data.json', log_dir='/mock/logs', log_file='mock_order_manager.log'):
        # Initialize the mock data file
        self.mock_data_file = mock_data_file
        self.orders = {}
        self.trades = []
        self.account_info = {}
        self.initialize_mock_data_file()
        self.mock_trade_file = 'src/mock_trading/mock_past_trades.json'
        self.mock_account_file = 'src/mock_trading/mock_account.json'

        # Initialize logging
        self.logger = self._initialize_logging(log_dir, log_file)
        self.logger.info("MockOrderManager initialized.")

    def _initialize_logging(self, log_dir, log_file):
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(os.path.join(log_dir, log_file))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def initialize_mock_data_file(self):
        """
        Create the mock data file if it doesn't exist, and initialize the structure.
        """
        if not os.path.exists(self.mock_data_file):
            self.logger.info(f"Mock data file not found. Creating {self.mock_data_file}.")
            self.reset_mock_data()
        else:
            with open(self.mock_data_file, 'r') as file:
                data = json.load(file)
                self.orders = data.get("orders", {})
                self.trades = data.get("trades", [])
                self.account_info = data.get("account_info", {})

    def reset_mock_data(self):
        """
        Reset the mock data file to a default state.
        """
        self.orders = {}
        self.trades = []
        self.account_info = {
            "balances": [{"asset": "USDT", "free": "100000.0", "locked": "0.0"}],
            "permissions": ["SPOT"]
        }
        self.save_mock_data()

    def save_mock_data(self):
        """
        Save the current state of orders, trades, and account info to the mock data file.
        """
        data = {
            "orders": self.orders,
            "trades": self.trades,
            "account_info": self.account_info
        }
        with open(self.mock_data_file, 'w') as file:
            json.dump(data, file, indent=4)
        self.logger.info("Mock data saved to file.")

    def get_current_price(self, symbol):
        try:
            response = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}')
            data = response.json()
            return float(data['price'])
        except Exception as e:
            self.logger.error(f"Error retrieving price for {symbol}: {e}")
            return 0.0
        
    def create_order(self, symbol, order_type, amount, price=-1):
        """
        Simulate creating an order.
        # for market order, the order is placed immediately
        """
        order_id = int(time.time())  # Mock order ID using timestamp
        timestamp = int(time.time() * 1000)  # Mock transaction time in milliseconds
        status = "FILLED" if price == -1 else "OPEN"

        order = {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": f"mock_{order_id}",  # Mock client order ID
            "transactTime": timestamp,
            "price": price if price != -1 else "Market Price",
            "origQty": amount,
            "executedQty": amount if status == "FILLED" else 0,
            "cummulativeQuoteQty": amount * price if price != -1 else 0,
            "status": status,
            "timeInForce": "GTC",
            "type": "MARKET" if price == -1 else "LIMIT",
            "side": order_type.upper()
        }

        # Update balances and trades
        price = self.get_current_price(symbol) if price == -1 else price
        self.update_mock_account(symbol, order_type, amount, price)
        if status == "FILLED":
            self.update_trade_file(symbol, order_type, amount, price)
        else:
            self.orders[order_id] = order

        self.save_mock_data()
        self.logger.info(f"Created mock order: {order}")
        return order

    def cancel_order(self, order_id):
        """
        Simulate cancelling an order.
        """
        if order_id in self.orders:
            order = self.orders.pop(order_id)
            order['status'] = "CANCELED"
            self.save_mock_data()
            self.logger.info(f"Cancelled mock order: {order}")
            return order
        else:
            self.logger.warning(f"Attempted to cancel non-existent order ID: {order_id}")
            return None

    def check_order_status(self, order_id):
        """
        Check the status of a mock order.
        """
        order = self.orders.get(order_id)
        if order:
            self.logger.info(f"Mock order status: {order}")
            return order
        else:
            self.logger.warning(f"No mock order found for ID: {order_id}")
            return None

    def get_account_info(self):
        """
        Simulate retrieving account info.
        """
        self.logger.info(f"Mock account info: {self.account_info}")
        try:
            with open(self.mock_account_file, 'r') as file:
                account_info = json.load(file)
            self.logger.info(f"Returning mock account info from file")
            return account_info
        
        except FileNotFoundError:
            self.logger.error(f"Mock account file {self.mock_account_file} not found.")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON in mock account file {self.mock_account_file}.")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_past_trades_from_api: {e}")
            return []

    def fetch_past_trades_from_api(self, symbol, limit=50):
        try:
            # Read the mock data file
            with open(self.mock_trade_file, 'r') as file:
                trades = json.load(file)

            # Filter trades for the specified symbol
            trades_for_symbol = [trade for trade in trades if trade["symbol"] == symbol]
            get_limit = min(limit, len(trades_for_symbol))
            limited_trades = trades_for_symbol[:get_limit]

            self.logger.info(f"Returning {get_limit} mock trades for {symbol} (limit: {limit}).")
            return limited_trades

        except FileNotFoundError:
            self.logger.error(f"Mock data file {self.mock_trade_file} not found.")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON in mock data file {self.mock_trade_file}.")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_past_trades_from_api: {e}")
            return []
        
    

    def update_mock_account(self, symbol, order_type, amount, price):
        """
        Update mock account balances based on order details.
        """
        base_asset = symbol[:-4]  # Assuming symbol is like BTCUSDT
        quote_asset = symbol[-4:]  # Assuming last 4 chars are the quote asset (e.g., USDT)

        # Read the current account info from the JSON file
        with open(self.mock_account_file, 'r') as file:
            account_info = json.load(file)

        # Convert string amounts to float for calculations
        balances = {balance['asset']: float(balance['free']) for balance in account_info['balances']}

        if order_type == "buy":
            cost = amount * (price if price != -1 else 1)  # Assume 1 for market price
            if balances.get(quote_asset, 0) >= cost:
                balances[quote_asset] -= cost
                balances[base_asset] = balances.get(base_asset, 0) + amount

        elif order_type == "sell":
            if balances.get(base_asset, 0) >= amount:
                balances[base_asset] -= amount
                balances[quote_asset] = balances.get(quote_asset, 0) + amount * (price if price != -1 else 1)

        # Update the account info with the new balances
        account_info['balances'] = [{"asset": asset, "free": str(free), "locked": "0.0"} for asset, free in balances.items()]

        # Write the updated account info back to the JSON file
        with open(self.mock_account_file, 'w') as file:
            json.dump(account_info, file, indent=4)

    def update_trade_file(self, symbol, order_type, amount, price):
        """
        Append a new trade to the mock trade file.
        """
        new_trade = {
            "symbol": symbol,
            "id": int(time.time()),  # Mock trade ID using timestamp
            "orderId": int(time.time()),  # Mock order ID using timestamp
            "side": order_type.upper(),
            "price": f"{price:.2f}",
            "qty": f"{amount:.8f}",
            "realizedPnl": "0.00",  # Assuming no realized PnL for simplicity
            "marginAsset": symbol[-4:],  # Assuming last 4 chars are the quote asset (e.g., USDT)
            "quoteQty": f"{amount * price:.2f}",
            "commission": "0.00",  # Assuming no commission for simplicity
            "commissionAsset": symbol[-4:],  # Assuming last 4 chars are the quote asset (e.g., USDT)
            "time": int(time.time() * 1000),  # Mock transaction time in milliseconds
            "positionSide": "BOTH",  # Assuming position side is BOTH for simplicity
            "buyer": order_type.lower() == "buy",
            "maker": False  # Assuming the order is not a maker order for simplicity
        }

        # Read the current trades from the JSON file
        try:
            with open(self.mock_trade_file, 'r') as file:
                trades = json.load(file)
        except FileNotFoundError:
            trades = []

        # Append the new trade
        trades.append(new_trade)

        # Write the updated trades back to the JSON file
        with open(self.mock_trade_file, 'w') as file:
            json.dump(trades, file, indent=4)
