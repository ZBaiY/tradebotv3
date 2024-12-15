"""
This is a distinct class for single symbol trading strategies.
For predicting from model and request risk manager for stop loss and take profit.
No need to import the BaseStrategy class here.
"""

# base_model.py
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_handling.real_time_data_handler import RealTimeDataHandler
from src.signal_processing.signal_processor import SignalProcessor, NonMemSignalProcessor
from src.models.base_model import ForTesting as TestModel
import src.models.ml_model as MLModel
import src.models.physics_model as PhysModel
import src.models.statistical_model as StatModel

"""
The risk manager as a instance in this class will only be used for 
to apply strategies that generate signals for the multiple assets.
Strategy class by it self will not do the risk management calculations.
"""

class SingleAssetStrategy:
    def __init__(self, symbol, m_config, d_config,risk_manager, data_handler, signal_processors, feature_extractor):
        """
        Strategy class for trading 
        a single asset.
        """
        self.symbol = symbol
        self.prediction = None
        # equity, balance, assigned_percentage are handled within risk managers
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        self.signal_processors = signal_processors
        self.feature_extractor = feature_extractor   
        self.model_type = m_config.get('method', None)
        self.params = m_config.get('params', {})

        self.decision_model = d_config.get('method', None)
        self.decision_params = d_config.get('params', {})
        self.decision_type = None
        self.decision_variant = None
        self.decision = {}
        


    def initialize(self, risk_manager):
        """
        Initialize strategy parameters.
        """
        self.risk_manager = risk_manager
        if not self.model_type:
            raise ValueError("Model type is not specified.")
        
        type_parts = self.model_type.split('_')
        model_category = type_parts[0]
        model_variant = type_parts[1]

        if model_category == 'ML':
            self.model = MLModel(self.symbol, self.data_handler, self.signal_processors, self.feature_extractor, model_variant, **self.params)
        elif model_category == 'Stat':
            self.model = StatModel(self.symbol, self.data_handler, self.signal_processors, self.feature_extractor, model_variant, **self.params)
        elif model_category == 'Phys':
            self.model = PhysModel(self.symbol, self.data_handler, self.signal_processors, self.feature_extractor, model_variant, **self.params)
        elif model_category == 'Test':
            self.model = TestModel(self.symbol, self.data_handler, self.signal_processors, self.feature_extractor, model_variant, **self.params)
            # print("Test model is used.")
            """Expand the model categories as the tradebot is developed."""
        else:
            raise ValueError(f"Unknown model category: {model_category}")
        
        type_parts = self.decision_model.split('_')
        self.decision_type = type_parts[0]
        self.decision_variant = type_parts[1]




    def run_prediction(self, data):
        self.prediction = self.model.predict(data, **self.params)
        self.risk_manager.request_prediction(self.prediction)
        self.risk_manager.calculate_stp()


    def make_decision_market(self):
        if self.decision_type == 'threshold':
            self.decision_params['current_price'] = self.request_data(self.data_handler, 'close')
            decis = DecesionMaker.threshold_based_decision(self.prediction, **self.decision_params)
        ########## Add more decision types here
        else: decis = "hold"
        self.decision['model_decis_forbacktest'] = decis

        if decis == "buy":
            self.decision['signal'] = "buy"
            capital = self.risk_manager.calculate_capital(buy=True)
            fraction = self.risk_manager.position_size
            price = self.data_handler.get_last_data(self.symbol)['close']
            self.decision['amount'] = fraction * capital/price
        elif decis == "sell":
            self.decision['signal'] = "sell"
            capital = self.risk_manager.calculate_capital(sell=True)
            fraction = self.risk_manager.position_size
            price = self.data_handler.get_last_data(self.symbol)['close']
            self.decision['amount'] = fraction * capital/price
        else:
            self.decision['signal'] = "hold"
            self.decision['amount'] = 0
            
        return self.decision
    
    def run_strategy_market(self, data):
        self.run_prediction(data)
        self.make_decision_market()
        

    def fit_model(self, data):
        """
        Fit the model on the given data.
        """
        self.model.fit(data, **self.params)


    def check_risk(self):
        """
        Check if the current risk level is within the acceptable range.
        """
        pass

    def set_equity(self, equity):
        self.equity = equity
    def set_balance(self, balance):
        self.balance = balance
    def set_assigned_percentage(self, assigned_percentage):
        self.assigned_percentage = assigned_percentage
    

    def request_data(self, datahandler, column):
        return datahandler.get_data(self.symbol)[column]
    def request_signal(self, signal_processor): # specify the signal processor
        return signal_processor.get_signal(self.symbol)
    def request_indicators(self, feature_extractor):
        return feature_extractor.get_indicators(self.symbol)

    def get_prediction(self):
        return self.prediction
    


class DecesionMaker:

    @staticmethod
    def threshold_based_decision(predictions, **kwargs):
        """
        Apply threshold-based decision logic.
        """
        current_price = kwargs.get('current_price', current_price)
        threshold = kwargs.get('threshold', threshold)
        if predictions[-1] > current_price * (1 + threshold):
            return "buy"
        elif predictions[-1] < current_price * (1 - threshold):
            return "sell"
        else:
            return "hold"
    