# Chronos Forecasting Docker

This Docker environment is designed to run and experiment with time series forecasting using **AutoGluon Chronos**. It includes example scripts for zero-shot forecasting, fine-tuning, and incorporating covariates into time series models.

## Files in This Docker

1. **`zero_shot_forecasting.py`**:  
   Demonstrates how to perform zero-shot forecasting using Chronos' pre-trained models.

2. **`fine_tuning_chronos.py`**:  
   Explains how to fine-tune Chronos models to improve forecasting accuracy for your specific time-series data.

3. **`incorporating_covariates.py`**:  
   Shows how to incorporate additional covariates (e.g., external variables) to enhance forecasting performance.

## How to Build and Run the Docker Container

### 1. Build the Docker Image
In the directory containing the `Dockerfile`, run:
docker build -t chronos_forecasting .

### 2. Run the Docker Container
To run the container and access a shell:
docker run --rm -it chronos_forecasting

### 3. Run a Specific Script
You can directly run any of the example scripts without opening a shell. For example:
docker run --rm -it chronos_forecasting python /app/zero_shot_forecasting.py

Stop and remove the container when you're done:
docker stop chronos_container
docker rm chronos_container

### Dependencies
The following Python dependencies are installed in the container:
autogluon.timeseries
pip, setuptools, and wheel (latest versions)