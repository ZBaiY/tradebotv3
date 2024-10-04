###### Updates needed: if last_fetch month is different, check the data files location, and move them (not very necessary)
###### Updates needed: detect market stops---volume==0
###### Updates needed: similar to the last one, check the validation of the data. But not for closing the bot

Explanation of cleaning json parameters
{
    "base_url": "https://api.binance.com",
    "endpoint": "/api/v3/klines",
    "frequency": "15m" ///for live data
}



{
    "params": {
        "check_labels": true,
        "clean": true,
        "resample_align": false, # the resample function is not used for history data 
        "timezome_adjust": false,
        "zero_variance": true,
        "remove_outliers": true,
        "rescale": true,
        "resample_freq": "h", #resemple frequency
        "outlier_threshold": 20, #outlier threshold
        "adjacent_count": 7, #for subtitude outliers, how many adjacent values to consider
        "utc_offset": 3, #UTC offset
        "scaler_type": "minmax" # methods to rescale data
    },
    "required_labels": [
        "open_time", "open", "high", "low", "close", "volume"
    ],
    "datetime_format": "ms"
}

# Load JSON configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Extract params and kwargs
params = config['params']
kwargs = {key: value for key, value in config.items() if key != 'params'}

# Example DataFrame
data = {
    'open_time': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'close_time': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'open': [1.0, 2.0, 3.0],
    'high': [1.5, 2.5, 3.5],
    'low': [0.5, 1.5, 2.5],
    'close': [1.2, 2.2, 3.2],
    'volume': [100, 200, 300]
}
df = pd.DataFrame(data)

# Initialize DataCleaner
df_cleaner = DataCleaner(df, params, **kwargs)

# Perform data type cleaning
df_cleaner.datatype_df()

# Print the cleaned DataFrame
print(data_cleaner.df)

df_cleaned = df_cleaner.get_cleaned_df():

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fetch_data.json:
{
  "symbol": "example_symbol",
  "interval": "example_interval",
  "start_date": "YYYY-MM-DD",
  "end_date": null, (or "YYYY-MM-DD")
  "limit": 500,
  "rate_limit_delay": 1
}

with open('fetch_data.json', 'r') as json_file:
        data = json.load(json_file)

    # Extract global limit and rate limit delay
    limit = data['limit']
    rate_limit_delay = data['rate_limit_delay']
    file_type = data['file_type']

    # Iterate over the symbols and intervals
    for symbol_data in data['symbols']:
        symbol = symbol_data['symbol']
        for interval_data in symbol_data['intervals']:
            interval = interval_data['interval']
            start_date = interval_data['start_date']
            end_date = interval_data['end_date']
            raw = interval_data['raw']
            rescaled = interval_data['rescaled']
            output_file = file_path(symbol, interval, start_date, end_date, raw=raw, rescaled=rescaled, file_type)
            prepare_data_chunks(symbol, interval, start_date, end_date, output_file, limit=limit, rate_limit_delay=rate_limit_delay, file_type)