import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from src.backtesting.backtester import SingleAssetBacktester, MultiAssetBacktester

if __name__ == '__main__':
    backtester_BTCUSDT = SingleAssetBacktester()
    backtester_BTCUSDT.run_initialization()
    backtester_BTCUSDT.run_backtest()
    equity_history = backtester_BTCUSDT.equity_history
    trade_history = backtester_BTCUSDT.trade_log
    history_to_csv = []
    for trade in trade_history:
        #{'symbol': self.symbol, 'date': self.current_date, 'price': price,
        #  'quantity': quantity, 'order': order, 'balance': self.balance, 'equity': self.equity}
        history_to_csv.append([trade['symbol'], trade['date'], trade['price'], trade['quantity'], trade['order'], trade['balance'], trade['equity']])
    trade_history = pd.DataFrame(history_to_csv, columns=['symbol', 'date', 'price', 'quantity', 'order', 'balance', 'equity'])
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtest', 'logs'))
    os.makedirs(output_path, exist_ok=True)
    trade_history.to_csv(os.path.join(output_path, 'trade_history_BTCUSDT.csv'))

    plt.plot(equity_history)
    plt.show()
