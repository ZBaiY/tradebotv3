import os
import sys

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    import scripts.backtest_v1
    import scripts.mock_trade
    import scripts.fetch_data

def test_backtest_main_runs():
    from scripts.backtest_v1 import main as backtest_main
    bt = backtest_main(symbol="BTCUSDT")
    assert hasattr(bt, "capital_full_position")
    assert len(bt.capital_full_position) > 0

def test_mock_trade_main_runs(monkeypatch):
    import src.mock_trading.mock_real_time_dealer as mrd
    monkeypatch.setattr(mrd.MockRealtimeDealer, "start", lambda self: None)
    from scripts.mock_trade import main as mock_main
    trader = mock_main()
    assert trader.__class__.__name__ == "MockRealtimeDealer"

if __name__ == "__main__":
    # Run the three tests manually
    test_imports()
    test_backtest_main_runs()
    # monkeypatch isn’t available outside pytest, so skip the last one
    print("Basic tests passed (manual run).")