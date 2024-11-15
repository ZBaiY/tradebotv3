from base_strategy import BaseStrategy

"""This Module will contains the prediction and models for multiple assets."""

class MultiAssetStrategy(BaseStrategy):
    def __init__(self, datahandler, risk_manager, feature_module=None, signal_processors=None):
        """
        Strategy class for trading
        multiple assets simultaneously.
        there might be multiple signal processors
        signal_processors: [signal_processor1, signal_processor2, ...]
        """
        super().__init__(datahandler, risk_manager, feature_module, signal_processors)
        self.current_data = {}
        self.processed_data = {}
        self.predictions = {}
        self.signals = {}
        self.strategies = {}
        
    def initialize_singles(self):
        for symbol in self.symbols:
            self.strategies[symbol] = ...
            self.strategies[symbol].initialize(self.risk_manager.risk_managers[symbol])
            self.strategies[symbol].set_equity(self.equity)
            self.strategies[symbol].set_balances(self.balances[symbol])
            self.strategies[symbol].set_assigned_capitals(self.assigned_calpitals[symbol])    

    def pre_run(self):
        rebanlance_need = self.check_rebalance()
        if rebanlance_need:
            self.rebalance_portfolio()

    def run_prediction(self, data):
        """
        Run prediction on the given data.
        """
        self.predictions = self.model.predict(data)
        self.pred_to_riskmanager()
        self.check_risk() ### This step can probably generate a signal
        self.apply_stp()
        self.generate_signals()

        """
        Reminder: There is request_data, etc. in the SingleAssetStrategy class, For 
        """



    def update_equity(self, equity):
        self.equity = equity
        for symbol in self.symbols:
            self.strategies[symbol].set_equity(equity)
    
    def update_balances(self, balances):
        self.balances = balances
        for symbol in self.symbols:
            self.strategies[symbol].set_balances(balances)
    
    def update_assigned_capitals(self, assigned_capitals):
        self.assigned_calpitals = assigned_capitals
        for symbol in self.symbols:
            self.strategies[symbol].set_assigned_capitals(assigned_capitals)

    def generate_signals(self):
        """
        Generate buy/sell signals based on predictions and processed data.
        """
        pass

    def pred_to_riskmanager(self):
        """Share prediction with RiskManager."""
        self.risk_manager.request_prediction(self.predictions)


    def apply_stp(self, trade_signal):
        """
        stp: stop loss and take profit and position size
        Adjust trade signal according to stop loss and take profit from portfolio manager.
        :param trade_signal: Signal to be adjusted
        """
        stop_loss, take_profit = self.risk_manager.calculate_stp()
        # Apply SL/TP logic to the trade_signal here

        
    def check_risk(self):
        """
        Check if the current risk level is within the acceptable range.
        """
        pass

        
