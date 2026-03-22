#Imports all necessary packages
try:
    from flask import Flask, request, jsonify
    from fmi_fetch import fetch_fmi_weather_data, FMI_UNITS
    from model.model import WeatherModel
    from model.scaler import Scaler
    from utils.preprocess import prepare_features
    from utils.plot import plot_weather_preds
    import torch
    import webbrowser
    import os
    from threading import Timer
    import numpy as np
except ImportError:
    #Raise error if packages are missing
    raise RuntimeError(
        "Missing Python packages.\n"
        "Run: pip install -r requirements.txt to install the necessary packages"
    )


#Initiate Flask app
app = Flask(__name__)

#Loads trained model and maps it to CPU
model = WeatherModel()
model.load_state_dict(torch.load('trained_model/weather_model.pth', map_location='cpu'))
model.eval()

#Loads scaler for normalization for input data
scaler = Scaler.load("trained_model/scaler.pth")

#Home site (simple HTML)
@app.route('/', methods=['GET'])
def home():
    return """Weather API is running! 
            <br><br> 
            Now start the Qt-client by running the command <code>python client\qt.py</code> in a separate terminal to see the visual interface. 
            <br><br> 
            You can even explore other Flask-endpoints, like <code>/about</code> and <code>/predict</code>, for example: http://127.0.0.1:5000/about"""

#Information site of the application
@app.route('/about', methods=['GET'])
def about():
    return jsonify({
        "model": "Neural Network (PyTorch)",
        "predicts": "Temperatures at one hour intervals in the future",
        "inputs": [
            "Current temperature",
            "Wind speed",
            "Wind direction (sin/cos)",
        ],
        "location": "Helsinki, Kaisaniemi"
    })

#Endpoint for fetching actual weather data
@app.route('/weather', methods=['GET'])
def get_weather():
    place = request.args.get('place', 'Helsinki')

    df = fetch_fmi_weather_data(place)

    #Returns latest 10 observations
    df_out = df.tail(10).copy()

    #Replaces NaN with None and valid JSON
    df_out = df_out.replace({np.nan: None})
    
    return jsonify({
        "units": FMI_UNITS, #Units for each column
        "data": df_out.to_dict(orient='records') #Data as lists of dicts
    })

#Endpoint for temperature predictions
@app.route('/predict', methods=['GET'])
def predict_weather():
    place = request.args.get('place', 'Helsinki')   

    #Fetches latest weather data
    df = fetch_fmi_weather_data(place)

    #Prepares features and normalizes with scaler
    X = prepare_features(df, scaler)

    #Makes predicitions without gradient calculation for saving memory and time
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten().tolist() #Sets predictions as list
        #Transforms to original scale
        preds = [p * scaler.temp_std + scaler.temp_mean for p in preds]

    #Puts predicitions into DataFrame
    df_out = df.tail(len(preds)).copy()
    df_out["temperature_prediction_next_hour"] = preds

    #Calculates absolute error
    df_out["abs_error"] = (
        df_out["temperature_prediction_next_hour"] - df_out["temperature"]
    ).abs()

    #Calculates MAE
    mae = float(df_out["abs_error"].mean())

    #Creates persistence-baseline by using previous observation
    df_out["baseline_persistence"] = df_out["temperature"].shift(1)
    baseline_mae = (
        df_out["baseline_persistence"] - df_out["temperature"]
    ).abs().mean()

    #Creates and saves graph of actual and predicted temperatures
    plot_path = plot_weather_preds(df_out)

    #Rounds some columns for better readability in JSON
    df_out = df_out.round({
        "abs_error": 3,
        "baseline_persistence": 1,
        "temperature": 2,
        "temperature_prediction_next_hour": 3,
        "humidity": 1,
        "pressure": 1,
        "winddir_sin": 4,
        "winddir_cos": 4
    })

    #Replaces NaN with None for JSON
    df_out = df_out.replace({np.nan: None})
    
    return jsonify({
        "meta": {
            "mae": mae,
            "mae_baseline": baseline_mae,
            "n_points": len(df_out)
        },
        "units": {
            **FMI_UNITS,
            "temperature_prediction_next_hour": "°C"
        },
        "data": df_out.to_dict(orient='records'),
        "plot": plot_path #Path to saved graph
    })

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/"

    #Opens browser automatically
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("Starting Flask in web browser...")
        Timer(1.0, lambda: webbrowser.open(url)).start()
    
    #Starts Flask server
    app.run(host="127.0.0.1", port=5000, debug=True)
