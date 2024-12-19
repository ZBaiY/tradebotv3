import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from src.backtesting.backtester import SingleAssetBacktester, MultiAssetBacktester


if __name__ == '__main__':
    backtester = MultiAssetBacktester()
    backtester.run_initialization()
    backtester.run_backtest()
    equity_history = backtester.equity_history
    trade_history = backtester.trade_log
    model_history = backtester.log_model
    model_to_csv = []
    history_to_csv = []
    for trade in trade_history:
        #{'symbol': self.symbol, 'date': self.current_date, 'price': price,
        #  'quantity': quantity, 'order': order, 'balance': self.balance, 'equity': self.equity}
        # {'symbol': self.symbol, 'date': self.current_date, 'price': price,'order': order}
        history_to_csv.append([trade['symbol'], trade['date'], trade['price'], trade['quantity'], trade['order'], trade['balance'], trade['equity']])
    for model in model_history:
        model_to_csv.append([model['symbol'], model['date'], model['price'], model['order']])
    trade_history = pd.DataFrame(history_to_csv, columns=['symbol', 'date', 'price', 'quantity', 'order', 'balance', 'equity'])
    model_history = pd.DataFrame(model_to_csv, columns=['symbol', 'date', 'price', 'order'])
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtest', 'logs'))
    os.makedirs(output_path, exist_ok=True)
    trade_history.to_csv(os.path.join(output_path, 'trade_history.csv'))
    model_history.to_csv(os.path.join(output_path, 'model_history.csv'))
    capital_history = pd.Series(backtester.capital_full_position)
    capital_history.to_csv(os.path.join(output_path, 'capital_history.csv'))

    plt.plot(equity_history)
    plt.show()
    pass

"""

if __name__ == '__main__':
    backtester_BTCUSDT = SingleAssetBacktester()
    backtester_BTCUSDT.run_initialization()
    backtester_BTCUSDT.run_backtest()
    equity_history = backtester_BTCUSDT.equity_history
    trade_history = backtester_BTCUSDT.trade_log
    model_history = backtester_BTCUSDT.log_model
    model_to_csv = []
    history_to_csv = []
    for trade in trade_history:
        #{'symbol': self.symbol, 'date': self.current_date, 'price': price,
        #  'quantity': quantity, 'order': order, 'balance': self.balance, 'equity': self.equity}
        # {'symbol': self.symbol, 'date': self.current_date, 'price': price,'order': order}
        history_to_csv.append([trade['symbol'], trade['date'], trade['price'], trade['quantity'], trade['order'], trade['balance'], trade['equity']])
    for model in model_history:
        model_to_csv.append([model['symbol'], model['date'], model['price'], model['order']])
    trade_history = pd.DataFrame(history_to_csv, columns=['symbol', 'date', 'price', 'quantity', 'order', 'balance', 'equity'])
    model_history = pd.DataFrame(model_to_csv, columns=['symbol', 'date', 'price', 'order'])
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtest', 'logs'))
    os.makedirs(output_path, exist_ok=True)
    trade_history.to_csv(os.path.join(output_path, 'trade_history_BTCUSDT.csv'))
    model_history.to_csv(os.path.join(output_path, 'model_history_BTCUSDT.csv'))
    capital_history = pd.Series(backtester_BTCUSDT.capital_full_position)
    capital_history.to_csv(os.path.join(output_path, 'capital_history_BTCUSDT.csv'))

    plt.plot(equity_history)
    plt.show()

"""