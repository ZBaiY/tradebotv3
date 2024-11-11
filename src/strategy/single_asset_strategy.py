"""
This is a distinct class for single symbol trading strategies.
For predicting from model and request risk manager for stop loss and take profit.
No need to import the BaseStrategy class here.
"""

class SingleAssetStrategy:
    def __init__(self, symbol, config):
        """
        Strategy class for trading 
        a single asset.
        """
        self.symbol = symbol
        self.prediction = None
        self.risk_manager = None
        self.model = config.get('model', None)
        self.params = config.get('params', {})


    def initialize(self, risk_manager):
        """
        Initialize strategy parameters.
        """
        self.risk_manager = risk_manager

    def request_data(self, datahandler, column):
        """
        Request data from the data handler.
        """
        return datahandler.get_data(self.symbol)[column]

    def request_signal(self, signal_processor):
        """
        Request signal from the signal processor.
        """
        return signal_processor.get_signal(self.symbol)
    
    def request_indicators(self, feature_extractor):
        """
        Request features from the feature extractor.
        """
        return feature_extractor.get_indicators(self.symbol)


    def fit_model(self, data):
        """
        Fit the model on the given data.
        """
        self.model.fit(data, **self.params)

    def run_prediction(self, data):
        """
        Run prediction on the given data.
        """
        self.prediction = self.model.predict(data, **self.params)

    def update_parameters(self, new_params):
        """
        Update model parameters.
        """
        self.params = new_params

    def get_prediction(self):
        """
        Get prediction for the given data.
        """
        return self.prediction
