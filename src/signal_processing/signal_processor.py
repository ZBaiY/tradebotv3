import pandas as pd
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
from src.signal_processing.filters import MovingAverageFilter, ExponentialSmoothingFilter
from src.signal_processing.transform import ReturnTransformer, LogReturnTransformer, ScalerTransformer, FourierTransformer

"""They only process one column at a time"""

class SignalProcessor:
    def __init__(self, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.data_handler.subscribe(self)  # Subscribe to the DataHandler
        self.symbols = self.data_handler.symbols
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.processed_data = {symbol: self.data_handler.cleaned_data[symbol][column].copy() for symbol in self.symbols}        
        self.lookback = 1
        self.k_space = False
        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = max(filter_instance.lookback, self.lookback)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            self.lookback = max(trans_instance.lookback, self.lookback)
        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rooling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        
    
    def apply_filters(self):
        for filter_instance in self.filters:
            for symbol in self.symbols:
                recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                self.processed_data[symbol] = filter_instance.apply(self.processed_data[symbol])
                self.processed_data[symbol] = self.processed_data[symbol].reindex(recent_timestamps)

    
    def apply_transform(self):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            for symbol in self.symbols:
                if not isinstance(trans_instance, FourierTransformer) and not k_space:
                    recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                    self.processed_data[symbol] = trans_instance.transform(self.processed_data[symbol])
                    self.processed_data[symbol] = self.processed_data[symbol].reindex(recent_timestamps)
                else:
                    self.processed_data[symbol] = trans_instance.transform(self.processed_data[symbol])
                    k_space = True

    def apply_all(self):
        """First filter the data, then apply transformations."""
        self.apply_filters()
        self.apply_transform()

        print(len(self.processed_data['BTCUSDT']))
        print(self.window_size)

    ####### Real Time Updating Functions ########


    def update(self, new_data):
        
        updated_window = {}
        for symbol in self.symbols:
            updated_window[symbol] = self.data_handler.get_data_limit(symbol, self.lookback+self.na_num, clean=True)
        for filter_instance in self.filters:
            for symbol in self.symbols:
                updated_window[symbol] = filter_instance.apply(updated_window[symbol])
                self.processed_data[symbol] = self.processed_data[symbol].append(updated_window[symbol].iloc[-1]).tail(self.window_size)






    """def update_filters(self, new_data, First = True):
       

        
        return First


    def update_transform(self, new_data , First = True): 
        
        Incrementally update the transformations based on new incoming data.
        Use new_data for efficient updates to avoid recalculating over the entire cleaned_data.
        
        :param new_data: dict, new incoming data for each symbol (pd.Series indexed by datetime)
        

        update_data = {}
        
        for trans_instance in self.trans:
            if First: # if first transformation, we will append new data
                if isinstance(trans_instance, ReturnTransformer) or isinstance(trans_instance, LogReturnTransformer):
                    for symbol in self.symbols:
                        This only updates one entry at a time so we append one and drop the oldest row.
                        two_data = self.data_handler.get_data_limit(symbol, 2, clean=True)
                        update_data[symbol] = trans_instance.on_new_data(two_data)
                        self.processed_data[symbol] = self.processed_data[symbol].append(update_data[symbol]).tail(self.window_size)
                        
                    First = False

                elif isinstance(trans_instance, ScalerTransformer):
                    for symbol in self.symbols:
                        update_data[symbol] = trans_instance.on_new_data(symbol, new_data[symbol])
                        self.processed_data[symbol] = self.processed_data[symbol].append(update_data[symbol]).tail(self.window_size)    
                        
                    First = False

            else:
                if isinstance(trans_instance, ReturnTransformer) or isinstance(trans_instance, LogReturnTransformer):
                    for symbol in self.symbols:
                        two_data = self.processed_data[symbol].tail(2)
                        update_data[symbol] = trans_instance.on_new_data(two_data)
                        self.processed_data[symbol].iat[-1] = update_data[symbol].iloc[-1]

                elif isinstance(trans_instance, ScalerTransformer):
                    for symbol in self.symbols:
                        data = self.processed_data[symbol].tail(1)
                        update_data[symbol] = trans_instance.on_new_data(symbol, data)
                        self.processed_data[symbol].iat[-1] = update_data[symbol].iloc[-1]"""
                        
            
