import os
import sys

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    import scripts.backtest_v1
    import scripts.mock_trade
    import scripts.fetch_data

def test_backtest_main_runs(monkeypatch):

    import src.backtesting.backtester as btmod
    # Skip the full 36k‑row loop
    monkeypatch.setattr(
        btmod.SingleAssetBacktester,
        "run_backtest",
        lambda self: None
    )

    from scripts.backtest_v1 import main as backtest_main
    bt = backtest_main(symbol="BTCUSDT")
    assert hasattr(bt, "run_initialization")

def test_mock_trade_main_runs(monkeypatch):
    import src.mock_trading.mock_real_time_dealer as mrd
    monkeypatch.setattr(mrd.MockRealtimeDealer, "start", lambda self: None)
    from scripts.mock_trade import main as mock_main
    trader = mock_main()
    assert trader.__class__.__name__ == "MockRealtimeDealer"

if __name__ == "__main__":
    """
    Run this file directly the same way `pytest` would, so fixture‑based tests
    (e.g., those using the `monkeypatch` fixture) still work without errors.
    """
    import pytest, sys
    sys.exit(pytest.main([__file__]))