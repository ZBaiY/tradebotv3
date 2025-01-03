import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from src.mock_trading.mock_real_time_dealer import MockRealtimeDealer

if __name__ == '__main__':
    trader = MockRealtimeDealer()
    # trader.run_initialization()
    trader.reset_config()
    trader.start()