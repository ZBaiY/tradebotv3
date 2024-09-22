Explanation of cleaning json parameters

{
    "params": {
        "check_labels": true,
        "clean": true,
        "resample_align": true,
        "timezome_adjust": false,
        "zero_variance": true,
        "remove_outliers": true,
        "rescale": true,
        "resample_freq": "H", #resemple frequency
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