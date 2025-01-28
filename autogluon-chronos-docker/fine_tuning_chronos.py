from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load the Australian electricity demand dataset
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)

# Split into train and test datasets
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Train and fine-tune Chronos-Bolt (small model)
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}}
        ]
    },
    time_limit=60,  # Limit training to 60 seconds
    enable_ensemble=False,
)

# Evaluate the trained models and generate the leaderboard
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
