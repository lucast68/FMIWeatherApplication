#Imports necessary packages
import numpy as np
import pandas as pd
import torch

#Function for preparing and normalizing weather data to tensor format that can be used in the neural network
#Creates cyclic representation of wind direction, normalizes variables with parameters and calculates historical statistic features
def prepare_features(df, scaler, history_days: int = 14):
    
    #Ensure: df can't be None
    if df is None:
        raise ValueError("prepare_features received None")

    #Controls that necessary columns exist
    required_basic = ["temperature", "wind_speed"]
    missing = [col for col in required_basic if col not in df.columns]
    if missing:
        raise ValueError(f"Följande kolumner saknas i DataFrame: {missing}")

    df = df.copy()

    #Ensures correct time index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.set_index('timestamp').sort_index()

    #Cyclic representation of wind direction (sin/cos)
    if 'winddir_sin' not in df.columns or 'winddir_cos' not in df.columns:
        if 'wind_direction' in df.columns:
            df['wind_direction'] = pd.to_numeric(
                df['wind_direction'], errors='coerce'
            ).fillna(0.0)

            df['winddir_sin'] = np.sin(np.radians(df['wind_direction']))
            df['winddir_cos'] = np.cos(np.radians(df['wind_direction']))
        else:
            df['winddir_sin'] = 0.0
            df['winddir_cos'] = 0.0
    
    #Controls that all columns exist and are numeric
    required_cols = ["temperature", "wind_speed", "winddir_sin", "winddir_cos", "humidity", "pressure"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    #Temperature change between two time steps
    df["temp_diff"] = df["temperature"].diff().fillna(0.0)

    #Normalization based on statistics from training data
    df["temperature"] = (df["temperature"] - scaler.temp_mean) / scaler.temp_std
    df["wind_speed"] = (df["wind_speed"] - scaler.wind_mean) / scaler.wind_std
    df["temp_diff"] = df["temp_diff"] / scaler.temp_diff_std
    df["humidity"] = df["humidity"] / 100.0
    df["pressure"] = (df["pressure"] - scaler.pressure_mean) / scaler.pressure_std

    #Calculation of historic rolling statistic values
    window = f"{history_days}D"
    agg_cols = ["temperature", "wind_speed", "humidity", "pressure"]
    for col in agg_cols:
        try:
            #Uses a time-based rolling window if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            roll = df[col].rolling(window=window, min_periods=1)
        except Exception:
            #Backup solution: Calculate rolling window based on amount of rows
            deltas = df.index.to_series().diff().dt.total_seconds().dropna()
            if len(deltas) == 0 or deltas.median() == 0:
                rows_per_day = 24  #Assumes hour data
            else:
                rows_per_day = 86400.0 / deltas.median()
            int_window = max(1, int(history_days * rows_per_day))
            roll = df[col].rolling(window=int_window, min_periods=1)
        
        df[f"{col}_hist_mean"] = roll.mean()
        df[f"{col}_hist_std"] = roll.std().fillna(0.0)
        df[f"{col}_hist_min"] = roll.min()
        df[f"{col}_hist_max"] = roll.max()

    #Collects feature columns used by the model
    feature_cols = [
        "temperature",
        "temp_diff",
        "wind_speed",
        "winddir_sin",
        "winddir_cos",
        "humidity",
        "pressure"
    ]

    for col in agg_cols:
        feature_cols += [f"{col}_hist_mean", f"{col}_hist_std", f"{col}_hist_min", f"{col}_hist_max"]

    #Replaces eventual NaN values
    df[feature_cols] = df[feature_cols].fillna(0.0)

    #Converts to PyTorch tensor
    tensors = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    return tensors
