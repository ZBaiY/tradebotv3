def test_imports():
    import scripts.backtest_v1 as bt
    import scripts.mock_trade as mt

def test_backtest_runs(tmp_path):
    from scripts.backtest_v1 import main  
    main(symbol="BTCUSDT", steps=1, data_dir=tmp_path)
"""
Basic smoke tests
=================

These tests only check that the high‑level scripts import and run without
crashing.  They are deliberately lightweight so they can execute inside the
Docker build / CI pipeline in a few seconds.
"""

def test_imports():
    """
    All top‑level scripts should import cleanly.
    """
    import scripts.backtest_v1
    import scripts.mock_trade
    import scripts.fetch_data


def test_backtest_main_runs():
    """
    `main()` in backtest_v1 should execute without raising and return a
    backtester instance that exposes an equity curve attribute.
    """
    from scripts.backtest_v1 import main as backtest_main

    bt = backtest_main(symbol="BTCUSDT")      # runs full back‑test
    assert hasattr(bt, "capital_full_position")
    # Ensure the equity curve is not empty
    assert len(bt.capital_full_position) > 0


def test_mock_trade_main_runs(monkeypatch):
    """
    Launch the mock trader in a way that terminates quickly.

    We monkey‑patch `MockRealtimeDealer.start` to avoid spinning a real loop.
    """
    import src.mock_trading.mock_real_time_dealer as mrd

    # Replace the infinite‑looping `start` with a no‑op.
    monkeypatch.setattr(mrd.MockRealtimeDealer, "start", lambda self: None)

    from scripts.mock_trade import main as mock_main
    trader = mock_main()

    # Returned object should be an instance of MockRealtimeDealer
    assert trader.__class__.__name__ == "MockRealtimeDealer"