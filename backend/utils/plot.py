#Uses matplotlib in "headless"-mode (without GUI)
#Graph gets saved as PNG-file and doesn't show up directly in the application
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd

#Creates temperature graph and both actual and predicted temperatures
#Returns path to the saved picture
def plot_weather_preds(df, out_path="static/prediction_plot.png"):
    #Makes the folder if it doesn't exist
    os.makedirs("static", exist_ok=True)

    #Ensures that timestamp exists and is in datetime-format
    if 'timestamp' in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    #Tries to read timestamps if the first attempt failed
    if df['timestamp'].isnull().all():
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), errors='coerce')

    #If timestamps are still missing index are used instead
    if df['timestamp'].isnull().all():
        x = range(len(df))
        use_dates = False
    else:
        x = df['timestamp']
        use_dates = True

    #Main configuration for figure
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10,10), dpi=150)

    #Draws actual temperature if column exists
    if 'temperature' in df.columns:
        plt.plot(df["timestamp"], df["temperature"], label="Actual temperature")
    
    #Draws predicted temperature if column exists
    pred_col = 'temperature_prediction_next_hour'
    if pred_col in df.columns:
        plt.plot(x, df[pred_col], label="Predicted temperature")

    #Formats time axis if date is used
    if use_dates:
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    #Axis categories and titles
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature vs Prediction")
    plt.legend()

    #Adapts layout and creates figure for a file
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    #Returns the absolute path for the picture
    return os.path.abspath(out_path)
