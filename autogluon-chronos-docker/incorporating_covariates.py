from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load the grocery sales dataset
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv"
)

# Split into train and test datasets
prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)

# Train Chronos-Bolt with and without covariates
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"]
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot model WITHOUT covariates
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            # Chronos-Bolt WITH covariates
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ]
    },
    enable_ensemble=False,
    time_limit=60,
)

# Evaluate the trained models and generate the leaderboard
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)
