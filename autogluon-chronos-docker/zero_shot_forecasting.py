from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load the Australian electricity demand dataset
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)

# Split into train and test datasets
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Train the predictor using Chronos-Bolt (small model) in zero-shot mode
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small"
)

# Generate and visualize predictions
predictions = predictor.predict(train_data)
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],  # Visualize the first two time series
    max_history_length=200,
)
