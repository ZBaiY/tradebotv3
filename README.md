# tradebotv3
The most exciting version, with real time operations
Update the scalers every one week
The costum selector, need to develop it for next version

# My Trading Bot Architecture

```mermaid
graph TD
    subgraph Real-Time Trading Bot
        A[RealTimeDataHandler] -->|Notifies new data| B[SignalProcessing]
        A -->|Notifies new data| C[Model]
        A -->|Notifies new data| D[Feature]
        
        B -->|Processes signals| E[Strategy]
        C -->|Produces predictions| E
        D -->|Extracts features| E

        E -->|Generates buy/sell signals| F[RealtimeDealer]
        A -->|Updates health and fetches data| F
    end
    
    F -->|Monitors system and manages data flow| A