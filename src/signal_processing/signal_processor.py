import pandas as pd

class SignalProcessor:
    def __init__(self, data_handler, filters=None, transform=None):
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
        self.processed_data = self.data_handler.cleaned_data.copy()
        self.window_size = self.data_handler.window_size


    def apply_filters(self):
        for filter_instance in self.filters:
            self.processed_data = filter_instance.apply(self.processed_data)
    
    def apply_transform(self):
        # Apply all filters to the data
        for trans_instance in self.trans:
            self.processed_data = trans_instance.transform(self.processed_data)

    def apply_all(self):
        self.apply_filters()
        self.apply_transform()

    def update_filters(self):
        """
        scalerfilter can do incremental fitting
        return can do semi-incremental fitting, with a window size of 2
        FFT cannot do incremental fitting
        """
        pass

    def update_transform(self):
        
        pass

    def update(self):
        self.update_filters()
        self.update_transform()