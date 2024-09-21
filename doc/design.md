The project is object-oriented. 
frequency and things like that can be object parameters 

class DataSettings:
    def -_-init__(self, frequency='1d', look_back_period=14):
        self.frequency = frequency
        self.look_back_period = look_back_period


or use
def calculate_returns(data, frequency='1d', look_back_period=14, **kwargs):
    # kwargs can accept additional optional parameters
    volatility = kwargs.get('volatility', None)  # Optional parameter
    pass




Structure:
tradebot_v3/
├── README.md
├── config/
│   ├── config.yaml
│   └── other_configs.json
├── data/
│   ├── processed/
│   ├── raw/
│   ├── rescaled/
│   ├── stats/
│   └── real_time/              # Folder for real-time data
├── notebooks/
├── reports/
├── requirements.txt
├── scripts/
│   ├── analyze_symbols.py
│   ├── deployment.py
│   ├── prepare_data.py
│   └── other_scripts.py
├── setup.py
└── src/
    ├── models/
    │   ├── base_model.py               # Defines BaseModel
    │   ├── statistical_model.py         # Inherits BaseModel
    │   ├── ml_model.py                  # Inherits BaseModel
    │   ├── physics_model.py             # Inherits BaseModel
    ├── strategy/
    │   ├── base_strategy.py             # Base class for all strategies
    │   ├── single_asset_strategy.py      # Inherits Strategy
    │   ├── multi_asset_strategy.py       # Inherits Strategy
    ├── portfolio_management/
    │   ├── portfolio_manager.py          # Defines PortfolioManager
    │   ├── capital_allocator.py          # Handles capital allocation
    │   ├── risk_manager.py               # Handles risk management
    │   ├── rebalancer.py                 # Rebalancing strategies
    ├── data_handling/
    │   ├── data_handler.py               # Defines DataHandler
    │   ├── historical_data_handler.py     # Manages historical data
    │   └── real_time_data_handler.py     # Manages real-time data
    ├── feature_engineering/
    │   ├── feature_extractor.py          # Defines feature extraction methods
    │   ├── feature_selector.py            # Handles feature selection methods
    ├── signal_processing/
    │   ├── signal_processor.py            # Defines SignalProcessor for preprocessing
    │   ├── filters.py                     # Signal filtering methods
    │   └── transform.py                   # Transformations like Fourier Transform
    ├── backtesting/
    │   ├── backtester.py                 # Backtesting engine
    │   └── performance_evaluation.py     # Performance metrics
    └── live_trading/
        ├── real_time_dealer.py           # Live trading handler
        ├── execution_handler.py           # Order execution logic
        └── order_manager.py               # Handles order statuses