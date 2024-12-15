import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.backtester import SingleAssetBacktester, MultiAssetBacktester

if __name__ == '__main__':
    backtester_BTCUSDT = SingleAssetBacktester()
    backtester_BTCUSDT.run_initialization()