import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from src.mock_trading.mock_real_time_dealer import MockRealtimeDealer

# -----------------------------------------------------------------------------  
# Public entry point for programmatic use (e.g. pytest, notebooks, other scripts)
# -----------------------------------------------------------------------------
def main():

    trader = MockRealtimeDealer()
    trader.reset_config()
    # trader.run_initialization()
    trader.start()
    return trader

if __name__ == "__main__":
    main()