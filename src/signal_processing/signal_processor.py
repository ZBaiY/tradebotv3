import pandas as pd
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_handling.real_time_data_handler import RealTimeDataHandler
from src.data_handling.historical_data_handler import HistoricalDataHandler, SingleSymbolDataHandler,MultiSymbolDataHandler
from src.signal_processing.filters import MovingAverageFilter, ExponentialSmoothingFilter
from src.signal_processing.transform import ReturnTransformer, LogReturnTransformer, ScalerTransformer, FourierTransformer, ScalerSymbolTransformer
import numpy as np
# from datetime import datetime, timezone
# import time

"""They only process one column at a time"""


"""
SignalProcessor class to process data signals.
And Store the processed data in the same format as the cleaned data.
The NoMemSignalProcessor is a class that does not store the processed data
but only returns it.
Remember to load the data_handler with the cleaned data before using the SignalProcessor.
"""
class SignalProcessor:
    def __init__(self, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        if isinstance(data_handler, RealTimeDataHandler):
            self.data_handler.subscribe(self)  # Subscribe to the DataHandler
        self.symbols = self.data_handler.symbols
        self.column = column
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.processed_data = {symbol: self.data_handler.cleaned_data[symbol][column].copy() for symbol in self.symbols}        
        self.lookback = 1
        self.k_space = False

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        
        
    def initilize_processors(self, config):
        print("initilize_processors")
        pass
        
    
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
                    if isinstance(trans_instance, ScalerTransformer):
                        trans_instance.fit_scaler(symbol, self.processed_data[symbol])
                        self.processed_data[symbol] = trans_instance.transform(symbol, self.processed_data[symbol])
                        self.processed_data[symbol] = self.processed_data[symbol].reindex(recent_timestamps)
                        continue
                    self.processed_data[symbol] = trans_instance.transform(self.processed_data[symbol])
                    self.processed_data[symbol] = self.processed_data[symbol].reindex(recent_timestamps)
                else:
                    self.processed_data[symbol] = trans_instance.transform(self.processed_data[symbol])
                    k_space = True

    def apply_all(self):
        """First filter the data, then apply transformations."""
        self.apply_filters()
        self.apply_transform()
    ####### Real Time Updating Functions ########


    def update(self, new_data):
        """Update the processed data with new incoming data."""
        """In most of cases, incrementally is not possible, due to the stacking of multiple filters and transformers."""
        self.processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}   
        self.apply_filters()
        self.apply_transform()     

    def get_signal(self, symbol):
        return self.processed_data[symbol]




class NonMemSignalProcessor:
    def __init__(self, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbols = self.data_handler.symbols
        self.column = column
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.lookback = 1
        self.k_space = False

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self, processed_data):
        for filter_instance in self.filters:
            for symbol in self.symbols:
                recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                processed_data[symbol] = filter_instance.apply(processed_data[symbol])
                processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
        return processed_data
    def apply_transform(self, processed_data):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            for symbol in self.symbols:
                if not isinstance(trans_instance, FourierTransformer) and not k_space:
                    recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                    if isinstance(trans_instance, ScalerTransformer):
                        trans_instance.fit_scaler(symbol, processed_data[symbol])
                        processed_data[symbol] = trans_instance.transform(symbol, processed_data[symbol])
                        processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
                        continue
                    processed_data[symbol] = trans_instance.transform(processed_data[symbol])
                    processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
                else:
                    processed_data[symbol] = trans_instance.transform(processed_data[symbol])
                    k_space = True
        return processed_data
    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data
    ####### Real Time Updating Functions ########


    def update(self, new_data):
        """Update the processed data with new incoming data."""
        """In most of cases, incrementally is not possible, due to the stacking of multiple filters and transformers."""
        processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data    
    
    def get_signal(self, symbol):
        return self.apply_all()[symbol]
    



class MemSymbolProcessor:
    def __init__(self, symbol, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbol = symbol
        self.column = column
        self.lookback = 1
        self.k_space = False
        self.processed_data = self.data_handler.cleaned_data[self.column].copy()

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self):
        for filter_instance in self.filters:
            recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
            self.processed_data = filter_instance.apply(self.processed_data)
            self.processed_data = self.processed_data.reindex(recent_timestamps)
    
    def apply_transform(self):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            if not isinstance(trans_instance, FourierTransformer) and not k_space:
                recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
                if isinstance(trans_instance, ScalerSymbolTransformer):
                    if not trans_instance.load:
                        trans_instance.fit_scaler(self.symbol, self.processed_data)
                    self.processed_data = trans_instance.transform(self.symbol, self.processed_data)
                    self.processed_data = self.processed_data.reindex(recent_timestamps)
                    continue
                self.processed_data = trans_instance.transform(self.processed_data)
                self.processed_data = self.processed_data.reindex(recent_timestamps)
            else:
                self.processed_data = trans_instance.transform(self.processed_data)
                k_space = True

    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        self.apply_filters()
        self.apply_transform()
    ####### Real Time Updating Functions ########

    def get_signal(self):
        return self.processed_data

    


class NonMemSymbolProcessor:
    def __init__(self, symbol, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbol = symbol
        self.column = column
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.lookback = 1
        self.k_space = False

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self, processed_data):
        for filter_instance in self.filters:
            recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
            processed_data = filter_instance.apply(processed_data)
            processed_data = processed_data.reindex(recent_timestamps)
        return processed_data
    
    def apply_transform(self, processed_data):
        # Apply all filters to the data
        k_space = False
        
        for trans_instance in self.trans:
            if not isinstance(trans_instance, FourierTransformer) and not k_space:
                recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
                if isinstance(trans_instance, ScalerSymbolTransformer):
                    if not trans_instance.load:
                        trans_instance.fit_scaler(self.symbol, processed_data)
                    processed_data = trans_instance.transform(self.symbol, processed_data)
                    processed_data = processed_data.reindex(recent_timestamps)
                    continue
                processed_data = trans_instance.transform(processed_data)
                processed_data = processed_data.reindex(recent_timestamps)
            else:
                processed_data = trans_instance.transform(processed_data)
                k_space = True
        return processed_data
    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        processed_data = self.data_handler.cleaned_data[self.symbol][self.column].copy()
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data
    ####### Real Time Updating Functions ########


    def update(self, new_data):
        """Update the processed data with new incoming data."""
        """In most of cases, incrementally is not possible, due to the stacking of multiple filters and transformers."""
        processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data    
    
    def get_signal(self):
        return self.apply_all()
    



class MemSymbolProcessor:
    def __init__(self, symbol, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbol = symbol
        self.column = column
        self.lookback = 1
        self.k_space = False
        self.processed_data = self.data_handler.cleaned_data[symbol][self.column].copy()

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self):
        for filter_instance in self.filters:
            recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
            self.processed_data = filter_instance.apply(self.processed_data)
            self.processed_data = self.processed_data.reindex(recent_timestamps)
    
    def apply_transform(self):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            if not isinstance(trans_instance, FourierTransformer) and not k_space:
                recent_timestamps = self.data_handler.cleaned_data[self.symbol].tail(self.window_size).index
                if isinstance(trans_instance, ScalerSymbolTransformer):
                    if not trans_instance.load:
                        trans_instance.fit_scaler(self.symbol, self.processed_data)
                    self.processed_data = trans_instance.transform(self.symbol, self.processed_data)
                    self.processed_data = self.processed_data.reindex(recent_timestamps)
                    continue
                self.processed_data = trans_instance.transform(self.processed_data)
                self.processed_data = self.processed_data.reindex(recent_timestamps)
            else:
                self.processed_data = trans_instance.transform(self.processed_data)
                k_space = True

    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        self.apply_filters()
        self.apply_transform()
    ####### Real Time Updating Functions ########

    def get_signal(self):
        return self.processed_data

    


class NonMemSymbolProcessorDataSymbol:
    def __init__(self, symbol, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbol = symbol
        self.column = column
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.lookback = 1
        self.k_space = False

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self, processed_data):
        for filter_instance in self.filters:
            recent_timestamps = self.data_handler.cleaned_data.tail(self.window_size).index
            processed_data = filter_instance.apply(processed_data)
            processed_data = processed_data.reindex(recent_timestamps)
        return processed_data
    
    def apply_transform(self, processed_data):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            if not isinstance(trans_instance, FourierTransformer) and not k_space:
                recent_timestamps = self.data_handler.cleaned_data.tail(self.window_size).index
                if isinstance(trans_instance, ScalerSymbolTransformer):
                    if not trans_instance.load:
                        trans_instance.fit_scaler(self.symbol, processed_data)
                    processed_data = trans_instance.transform(self.symbol, processed_data)
                    processed_data = processed_data.reindex(recent_timestamps)
                    continue
                processed_data = trans_instance.transform(processed_data)
                processed_data = processed_data.reindex(recent_timestamps)
            else:
                processed_data = trans_instance.transform(processed_data)
                k_space = True
        return processed_data
    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        processed_data = self.data_handler.cleaned_data[self.column].copy()        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data
    ####### Real Time Updating Functions ########


    def update(self, new_data):
        """Update the processed data with new incoming data."""
        """In most of cases, incrementally is not possible, due to the stacking of multiple filters and transformers."""
        processed_data = self.data_handler.cleaned_data[self.column].copy()    
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data    
    
    def get_signal(self):
        return self.apply_all()
    
class NonMemSignalProcessor:
    def __init__(self, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbols = self.data_handler.symbols
        self.column = column
        # Initialize the processed data with the cleaned data, it has the same window size as the cleaned data
        self.lookback = 1
        self.k_space = False

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self, processed_data):
        for filter_instance in self.filters:
            for symbol in self.symbols:
                recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                processed_data[symbol] = filter_instance.apply(processed_data[symbol])
                processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
        return processed_data
    def apply_transform(self, processed_data):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            for symbol in self.symbols:
                if not isinstance(trans_instance, FourierTransformer) and not k_space:
                    recent_timestamps = self.data_handler.cleaned_data[symbol].tail(self.window_size).index
                    if isinstance(trans_instance, ScalerTransformer):
                        trans_instance.fit_scaler(symbol, processed_data[symbol])
                        processed_data[symbol] = trans_instance.transform(symbol, processed_data[symbol])
                        processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
                        continue
                    processed_data[symbol] = trans_instance.transform(processed_data[symbol])
                    processed_data[symbol] = processed_data[symbol].reindex(recent_timestamps)
                else:
                    processed_data[symbol] = trans_instance.transform(processed_data[symbol])
                    k_space = True
        return processed_data
    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data
    ####### Real Time Updating Functions ########


    def update(self, new_data):
        """Update the processed data with new incoming data."""
        """In most of cases, incrementally is not possible, due to the stacking of multiple filters and transformers."""
        processed_data = {symbol: self.data_handler.cleaned_data[symbol][self.column].copy() for symbol in self.symbols}        
        processed_data = self.apply_filters(processed_data)
        processed_data = self.apply_transform(processed_data)
        return processed_data    
    
    def get_signal(self, symbol):
        return self.apply_all()[symbol]
    



class MemSymbolProcessorDataSymbol:
    def __init__(self, symbol, data_handler, column, filters=None, transform=None):
        """
        SignalProcessor class to process data signals.
        :filter: list, a list of filters to apply to the data, in order
        :transform: list, a list of transformations to apply to the data, in order
        """
        self.data_handler = data_handler
        if self.data_handler.window_size is None:
            print("Please load the data handler with the cleaned data before using the SignalProcessor")
            raise Exception("Data handler not loaded")
        self.filters = filters if filters is not None else []
        self.trans = transform if transform is not None else []
        self.symbol = symbol
        self.column = column
        self.lookback = 1
        self.k_space = False
        self.processed_data = self.data_handler.cleaned_data[self.column].copy()

        self.na_num = 0
        for filter_instance in self.filters:
            self.na_num += filter_instance.rolling_window-1 
        self.window_size = self.data_handler.window_size-self.na_num

        for filter_instance in self.filters:
            if filter_instance.lookback == 'all':
                self.lookback = self.window_size
            else: self.lookback = min(max(filter_instance.lookback, self.lookback), self.window_size)
        for trans_instance in self.trans:
            if trans_instance.lookback == 'all':
                self.lookback = self.window_size
                if isinstance(trans_instance, FourierTransformer):
                    self.k_space = True
            else: self.lookback = min(max(trans_instance.lookback, self.lookback), self.window_size)
            """if isinstance(trans_instance, ScalerTransformer):
                for symbol in self.symbols:
                    trans_instance.fit_scaler(symbol, self.processed_data[symbol])"""
        

    def apply_filters(self):
        for filter_instance in self.filters:
            recent_timestamps = self.data_handler.cleaned_data.tail(self.window_size).index
            self.processed_data = filter_instance.apply(self.processed_data)
            self.processed_data = self.processed_data.reindex(recent_timestamps)
    
    def apply_transform(self):
        # Apply all filters to the data
        k_space = False
        for trans_instance in self.trans:
            if not isinstance(trans_instance, FourierTransformer) and not k_space:
                recent_timestamps = self.data_handler.cleaned_data.tail(self.window_size).index
                if isinstance(trans_instance, ScalerSymbolTransformer):
                    if not trans_instance.load:
                        trans_instance.fit_scaler(self.symbol, self.processed_data)
                    self.processed_data = trans_instance.transform(self.symbol, self.processed_data)
                    self.processed_data = self.processed_data.reindex(recent_timestamps)
                    continue
                self.processed_data = trans_instance.transform(self.processed_data)
                self.processed_data = self.processed_data.reindex(recent_timestamps)
            else:
                self.processed_data = trans_instance.transform(self.processed_data)
                k_space = True

    
    def apply_all(self):
        """First filter the data, then apply transformations."""
        self.apply_filters()
        self.apply_transform()
    ####### Real Time Updating Functions ########

    def get_signal(self):
        return self.processed_data



if __name__ == "__main__":
    data_handler = RealTimeDataHandler('config/source.json', 'config/fetch_real_time.json')
    mv_filter = MovingAverageFilter(lookback_size=5)
    es_filter = ExponentialSmoothingFilter(alpha=0.3)
    filters = [mv_filter, es_filter]
    rt_trans = LogReturnTransformer()
    sclar_trans = ScalerTransformer(symbols=data_handler.symbols, scaler='minmax')
    fft = FourierTransformer()
    transform = [rt_trans, sclar_trans,fft]
    next_fetch_time,last_fetch_time = data_handler.pre_run_data()
    # # for symbol in data_handler.symbols:
    # #     sclar_trans.fit_scaler(symbol, data_handler.cleaned_data[symbol]['close'])
    # # print(data_handler.cleaned_data['BTCUSDT'].head())
    # # Pause for debugging
    # print("tap enter to continue...")
    # input()
    processors = SignalProcessor(data_handler, 'close', filters=filters, transform=transform)
    processors.apply_all()
    is_running = True
    while is_running:
        new_data = data_handler.data_fetch_loop(next_fetch_time, last_fetch_time)
        now = datetime.now(timezone.utc)
        data_handler.notify_subscribers(new_data)
        next_fetch_time = data_handler.calculate_next_grid(now)
        print(processors.processed_data['BTCUSDT'].tail())
        sleep_duration = (next_fetch_time - now).total_seconds()
        print(f"Sleeping for {sleep_duration} seconds until {next_fetch_time}")
        time.sleep(sleep_duration)
