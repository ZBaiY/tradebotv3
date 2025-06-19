# Tradebot v3

## Overview
The most exciting version, with real time operations
Update the scalers every one week
The costum selector, need to develop it for next version

Tradebot v3 is a multi-symbol crypto trading bot that automates data collection, strategy backtesting, and live trading using the Binance API. This repository includes code, documentation, and resources necessary to run, test, and further develop the bot.

## Purpose
- **Project Goal:** Automate crypto trading strategies using a combination of classical technical indicators and machine learning models.
- **Key Features:** 
  - Backtesting engine with detailed performance reporting.
  - Real-time data handling for live or mock trading.
  - Modular design for strategy evaluation and portfolio management.

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages can be installed via:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Code

| Mode | Local (Python) | Docker |
|------|----------------|--------|
| **Back-test** | `python scripts/backtest_v1.py` | `docker compose up backtester` |
| **Mock trade** | `python scripts/mock_trade.py` | `docker compose up mocktrader` |
| **Fetch data** | `python scripts/fetch_data.py` | `docker compose up fetcher` |

_Back-test results are written to `/backtest/performance`, mock-trade logs to `/mock/logs`, fetched raw data to `/data`._

### Progress-bar behaviour in Docker

When you launch services with `docker compose up`, the `tqdm` progress bar  
doesn’t have access to a real TTY, so each refresh prints on a new line.  
You have two options:

| Goal | Command |
|------|---------|
| **Clean, single‑line bar** | `docker compose run --rm --tty backtester` |
| **Quiet logs (no bar)** | `TQDM_DISABLE=1 docker compose up backtester` |

In standard local runs (`python scripts/backtest_v1.pNo source‑code changes are required.

- **Customize the configurations**
  For Backtest, go to: /backtest/config
  For Live/Mock Trading, go to /config
  There are some extra configurations for the mock accounts, go to /mock/config

## Pre-Analysis
- Folder 'notebooks' contains data manipulations and some tests for strategies before implementation

## Future Developments
- For planned enhancements and feature requests, refer to the `doc/Future developments.md` file.


## Contact
For questions or further information, please contact:  
Zhaoyu Bai – zbaiy.imsoion@yahoo.com


for DOGE or others who only works with integer amount, need to round down, we did it in risk manager, but also need to integrate it when dealing with fees.

## My Trading Bot Architecture

```mermaid
graph TD
    %% Real-Time Data Handling
    A[RealTimeDataHandler]
    A -->|Provides Data| C[Model]
    A -->|Provides Data| B[SignalProcessing]
    A -->|Provides Data| D[Feature Extraction]
    A -->|Updates health and fetches data| F[RealtimeDealer]

    %% Model & Signal Processing
    B -->|Provides processed data| C
    %% Strategy
    C -->|Provides predictions| E[Strategy]
    B -->|Processes signals for| E
    D -->|Supplies features to| E

    %% Risk Manager
    subgraph RiskManagement
        D -->|Provides features| G[RiskManager]
        E -->|Consults and listens to stop loss/take profit etc.| G
        C -->|Providese predictions| G
    end

    %% Strategy Integration with Risk Manager
    G -->|Provides risk guidelines| F
    E -->|Generates buy/sell signals| F

    %% Trade Execution Loop
    F -->|Executes trades in real-time| A```