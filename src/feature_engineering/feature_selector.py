import pandas as pd
import numpy as np

class TopSelector:
    def __init__(self, indicators, backtest_results):
        """
        :param indicators: A dictionary of available indicators.
        :param backtest_results: A DataFrame of backtesting results with a column 'performance'.
        """
        self.indicators = indicators
        self.backtest_results = backtest_results

    def select_indicators(self, top_n=3):
        """
        Select top N indicators based on past performance.
        :param top_n: Number of top indicators to select.
        :return: List of selected indicator names.
        """
        ranked_indicators = self.backtest_results.sort_values('performance', ascending=False)
        selected_indicators = ranked_indicators.head(top_n).index.tolist()
        return [self.indicators[indicator] for indicator in selected_indicators]


class CorrelationSelector:
    def __init__(self, indicator_data, threshold=0.7):
        """
        :param indicator_data: A DataFrame where each column represents the data of an indicator.
        """
        self.indicator_data = indicator_data
        self.threshold = threshold

    def select_indicators(self):
        """
        Select indicators that have a correlation less than the threshold.
        :param threshold: The maximum allowed correlation between indicators.
        :return: List of selected indicator names.
        """
        corr_matrix = self.indicator_data.corr()
        selected_indicators = []
        for i, indicator in enumerate(corr_matrix.columns):
            if not any(np.abs(corr_matrix.iloc[i, :i]) > self.threshold):
                selected_indicators.append(indicator)
        return selected_indicators
    
class VolatilityBasedSelector:
    def __init__(self, volatility, indicators):
        """
        :param volatility: Current market volatility.
        :param indicators: Dictionary of available indicators.
        """
        self.volatility = volatility
        self.indicators = indicators

    def select_indicators(self):
        """
        Select indicators based on current volatility.
        :return: List of selected indicator names.
        """
        if self.volatility > 0.02:  # Example threshold for high volatility
            return ['MACD', 'RSI']  # Momentum indicators for high volatility
        else:
            return ['SMA', 'EMA']  # Trend-following indicators for low volatility


#### The costum selector, need to develop it for next version
class HybridSelector:
    def __init__(self, backtest_results, indicator_data):
        """
        :param backtest_results: DataFrame with backtesting results.
        :param indicator_data: DataFrame with historical indicator data.
        """
        self.backtest_results = backtest_results
        self.indicator_data = indicator_data

    def select_indicators(self, top_n=3, corr_threshold=0.7):
        """
        Select top N indicators based on performance, ensuring low correlation.
        :param top_n: Number of top indicators to select.
        :param corr_threshold: Correlation threshold for selection.
        :return: List of selected indicator names.
        """
        top_indicators = self.backtest_results.sort_values('performance', ascending=False).head(top_n).index.tolist()
        corr_matrix = self.indicator_data[top_indicators].corr()

        selected_indicators = []
        for i, indicator in enumerate(corr_matrix.columns):
            if not any(np.abs(corr_matrix.iloc[i, :i]) > corr_threshold):
                selected_indicators.append(indicator)
        
        return selected_indicators
