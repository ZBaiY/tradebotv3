# tradebotv3
The most exciting version, with real time operations
Update the scalers every one week
The costum selector, need to develop it for next version

for DOGE or others who only works with integer amount, need to round down, we did it in risk manager, but also need to integrate it when dealing with fees.

# My Trading Bot Architecture

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
    F -->|Executes trades in real-time| A