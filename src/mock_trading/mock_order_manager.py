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

fee = 0.001


class MockOrderManager:
    def __init__(self, log_dir='../../mock/logs', log_file='mock_order_manager.log'):
        # Initialize the mock data file
        self.orders = {} # unexecuted orders
        self.account_info = {}
        
        self.mock_trade_file = 'mock/config/mock_past_trades.json'
        self.mock_account_file = 'mock/config/mock_account.json'
        self.mock_order_file = 'mock/config/mock_orders.json'
        self.logger = self._initialize_logging(log_dir, log_file)
        self.logger.info("MockOrderManager initialized.")
        
        self.initialize_mock_data_file()

        # Initialize logging
        

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
        if not os.path.exists(self.mock_account_file):
            self.logger.info(f"Mock account file {self.mock_account_file} not found. Creating new file.")
            with open(self.mock_account_file, 'w') as file:
                json.dump(self.account_info, file, indent=4)
        else:
            # print("account file exists")
            self.account_info = self.get_account_info()
            # print(self.account_info)
            # input('check order manager 52')
        """
        ##### no need to load in memories, one would read directly.
        if not os.path.exists(self.mock_trade_file):
            self.logger.info(f"Mock trade file {self.mock_trade_file} not found. Creating new file.")
            with open(self.mock_trade_file, 'w') as file:
                json.dump([], file, indent=4)
                
        else:
            with open(self.mock_trade_file, 'r') as file:
                self.trades = json.load(file)
            self.logger.info(f"Loaded {len(self.trades)} mock trades from file.")
        
        ##### no need to load in memories, one would read directly.

        if not os.path.exists(self.mock_order_file):
            self.logger.info(f"Mock order file {self.mock_order_file} not found. Creating new file.")
            with open(self.mock_order_file, 'w') as file:
                json.dump({}, file, indent=4)
        else:
            with open(self.mock_order_file, 'r') as file:
                self.orders = json.load(file)
            self.logger.info(f"Loaded {len(self.orders)} mock orders from file.")
        """

    


    def save_mock_data(self):
        pass

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
        id_symbol = {'BTCUSDT': 1, 'ETHUSDT': 2, 'BNBUSDT': 3, 'ADAUSDT': 4, 'DOGEUSDT': 5, 'SOLUSDT': 6, 'DOTUSDT': 7, 'UNIUSDT': 8, 'LTCUSDT': 9}
        order_id = int(time.time())*10+id_symbol[symbol]  # Mock order ID using timestamp
        timestamp = int(time.time() * 1000)  # Mock transaction time in milliseconds
        status = "FILLED" if price == -1 else "OPEN"

        order = {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": f"mock_{order_id}",  # Mock client order ID
            "transactTime": timestamp,
            "price": price,
            "origQty": amount,
            "executedQty": amount if status == "FILLED" else 0,
            "cummulativeQuoteQty": amount * price if price != -1 else 0,
            "status": status,
            "timeInForce": "GTC",
            "type": "MARKET" if price == -1 else "LIMIT",
            "side": order_type # buy or sell
        }

        # Update balances and trades
        price = self.get_current_price(symbol) if price == -1 else price
        # 因交易时间差，可能会导致价格变动。注意，需要对amount做一定的round down处理，以避免花更多的钱买入。
        order['price'] = price
        if status == "FILLED": # if market order
            self.update_trade_file(order)
            # input('check order manager 123, trade file')
            self.update_mock_account(order)
            # input('check order manager 125, account file')
        else:
            self.orders[order_id] = order
            self.locking_assets(order) # lock assets if order is not filled


        self.logger.info(f"Created mock order: {order}")
        return order
    
    def locking_assets(self, order):
        pass
    def release_assets(self, order):
        pass
    
    def update_mock_account(self, order):
        """
        Update mock account balances based on order details.
        """
        symbol = order["symbol"]
        order_type = order["side"]
        amount = float(order["executedQty"])
        price = float(order["price"])
        base_asset = symbol[:-4]  # Assuming symbol is like BTCUSDT
        quote_asset = symbol[-4:]  # Assuming last 4 chars are the quote asset (e.g., USDT)

        # Read the current account info from the JSON file
        with open(self.mock_account_file, 'r') as file:
            account_info = json.load(file)

        # Convert string amounts to float for calculations
        balances = {balance['asset']: float(balance['free']) for balance in account_info['balances']}
        locked = {balance['asset']: balance['locked'] for balance in account_info['balances']}
        if order_type == "buy":
            cost = amount * (price if price != -1 else 1)  # Assume 1 for market price
            if balances.get(quote_asset, 0) >= cost:
                balances[quote_asset] -= cost * (1+fee)
                balances[base_asset] = balances.get(base_asset, 0) + amount
            else:
                self.logger.warning(f"Insufficient balance to buy {amount} {base_asset} at {price} {quote_asset} each.")

        elif order_type == "sell":
            if balances.get(base_asset, 0) >= amount:
                balances[base_asset] -= amount
                balances[quote_asset] = balances.get(quote_asset, 0) + amount * (1-fee) * (price if price != -1 else 1)
            else:
                self.logger.warning(f"Insufficient balance to sell {amount} {base_asset} at {price} {quote_asset} each.")

        # Update the account info with the new balances
        account_info['balances'] = [{"asset": asset, "free": str(free), "locked": locked[asset]} for asset, free in balances.items()]
        self.account_info = account_info
        # Write the updated account info back to the JSON file
        with open(self.mock_account_file, 'w') as file:
            json.dump(account_info, file, indent=4)
    

    def execute_order(self):
        for order in self.orders:
            if order["status"] == "OPEN":
                symbol = order["symbol"]
                side = order["side"]
                market_price = self.get_current_price(symbol)
                if side == 'buy' and market_price <= order["price"]:
                    quantity = order["origQty"]
                    order['status'] = 'Filled'
                    self.update_mock_account(order)
                    self.update_trade_file(order)
                    self.release_assets(order)
                    self.orders.pop(order["orderId"])
                    # input('check order manager 174')

                    
                elif side == 'sell' and market_price >= order["price"]:
                    quantity = order["origQty"]
                    order['status'] = 'Filled'
                    self.update_mock_account(order)
                    self.update_trade_file(order)
                    self.release_assets(order)
                    self.orders.pop(order["orderId"])
                    # input('check order manager 184')

    def update_trade_file(self, trade):
        """
        Append a new trade to the mock trade file.
        """
        # Read the current trades from the JSON file
        try:
            with open(self.mock_trade_file, 'r') as file:
                trades = json.load(file)
        except FileNotFoundError:
            trades = []

        # Append the new trade
        if trade['status'] == "FILLED":
            trades.append(trade)
        else: print("Tried to write in but order not filled")
        # Write the updated trades back to the JSON file
        with open(self.mock_trade_file, 'w') as file:
            json.dump(trades, file, indent=4)
    

    def calculate_entry_price(self, symbol):
    # Reverse the list of trades to process anti-chronologically
        if os.path.exists(self.mock_trade_file):
            with open(self.mock_trade_file, 'r') as file:
                trades = json.load(file)
        trades_reversed = trades[::-1]
        
        total_qty = 0.0  # Total quantity of asset bought
        total_cost = 0.0  # Total cost of the asset (price * qty)

        for trade in trades_reversed:
            # Filter trades by symbol and side
            if trade["symbol"] == symbol and trade["side"] == "buy":
                price = float(trade["price"])
                qty = float(trade["qty"])

                # Accumulate total cost and quantity
                total_qty += qty
                total_cost += price * qty

        # Calculate entry price (if total_qty is zero, avoid division by zero)
        if total_qty == 0:
            return None  # No BUY trades for the given symbol
        entry_price = total_cost / total_qty
        return entry_price

    def cancel_order(self, order_id):
        """
        Simulate cancelling an order.
        """
        if order_id in self.orders:
            
            order = self.orders.pop(order_id)
            input('check order manager 238')
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
        
    

    def update_trades(self, new_trade):
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

    def reset_config(self):

        default_account_info = {
            "makerCommission": 15,
            "takerCommission": 15,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": 1624362346000,
            "accountType": "SPOT",
            "balances": [
                {
                    "asset": "BTC",
                    "free": "0.0000000",
                    "locked": "0.0000000"
                },
                {
                    "asset": "ETH",
                    "free": "0.0000000",
                    "locked": "0.0000000"
                },
                {
                    "asset": "USDT",
                    "free": "1000.00000000",
                    "locked": "0.00000000"
                }
            ],
            "permissions": ["SPOT"]
        }

        # Reset the mock account file
        with open(self.mock_account_file, 'w') as file:
            json.dump(default_account_info, file, indent=4)
        self.logger.info(f"Reset mock account file {self.mock_account_file}.")
        default_trades = [
            {
                "symbol": "ETHUSDT",
                "orderId": 17358143672,
                "clientOrderId": "mock_17358143672",
                "transactTime": 1735814367610,
                "price": 3468.000,
                "origQty": 0.14400031387882295,
                "executedQty": 0.14400031387882295,
                "cummulativeQuoteQty": 0,
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "MARKET",
                "side": "buy"
            },
            {
                "symbol": "ETHUSDT",
                "orderId": 17358143672,
                "clientOrderId": "mock_17358143672",
                "transactTime": 1735814367610,
                "price": 3468.000,
                "origQty": 0.14400031387882295,
                "executedQty": 0.14400031387882295,
                "cummulativeQuoteQty": 0,
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "MARKET",
                "side": "sell"
            }
        ]
        # Reset the mock trade file
        with open(self.mock_trade_file, 'w') as file:
            json.dump(default_trades, file, indent=4)
        self.logger.info(f"Reset mock trade file {self.mock_trade_file}.")

        # Reset the mock order file
        with open(self.mock_order_file, 'w') as file:
            json.dump({}, file, indent=4)
        self.logger.info(f"Reset mock order file {self.mock_order_file}.")
