# Use a lightweight Python base image with version 3.9
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python example scripts into the container
COPY zero_shot_forecasting.py /app/zero_shot_forecasting.py
COPY fine_tuning_chronos.py /app/fine_tuning_chronos.py
COPY incorporating_covariates.py /app/incorporating_covariates.py

# Update pip and install required dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir autogluon.timeseries

# Set a default command for the container (optional)
# You can run the examples manually if needed, so we'll just start a shell by default
CMD ["bash"]
