#Imports packages for HTTP, XML-parsing and data processing
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

#Maps FMI-parameters to units
FMI_UNITS = {
    "temperature": "°C",
    "wind_speed": "m/s",
    "wind_direction": "°",
    "humidity": "%",
    "pressure": "hPa"
}

#Fetches data from FMI's API for a specific location (e.g. Helsinki, Kaisaniemi)
#Fetches parameters: temperature, wind speed 10 min, wind direction 10 min, wind humidity and sea pressure
#'timestep' gives time interval (in minutes) between observations and in data.
def fetch_fmi_weather_data(place="Kaisaniemi", debug=True):
    BASE_URL = (
        "https://opendata.fmi.fi/wfs?"
        "service=WFS&version=2.0.0&request=GetFeature&"
        "storedquery_id=fmi::observations::weather::timevaluepair&"
        "parameters=t2m,ws_10min,wd_10min,rh,p_sea&"
        "timestep=60&"
    )

    #Sends GET-requests to FMI
    params = {"place": place}
    response = requests.get(BASE_URL, params=params, timeout=10) 

    #Controls status on HTTP-responses
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        body = response.text[:1000] if response.text else ""
        raise ValueError(f"Fel vid hämtning av data från FMI: {response.status_code}\nURL: {response.url}\nBody: {body}") from e

    if debug:
        print("Request URL:", response.url)
        print(response.text[:2000]) #Shows XML-responses in debugging

    #Namespace definitions for XML-parsing
    ns = {
        'wfs': 'http://www.opengis.net/wfs/2.0',
        'gml': 'http://www.opengis.net/gml/3.2',
        'wml2': 'http://www.opengis.net/waterml/2.0',
        'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0',
        'om': 'http://www.opengis.net/om/2.0',
        'xlink': 'http://www.w3.org/1999/xlink'
    }

    #Maps FMI parameters to DataFrame column names
    series_map = {
        "t2m": "temperature",
        "ws_10min": "wind_speed",
        "wd_10min": "wind_direction",
        "rh": "humidity",
        "p_sea": "pressure"
    }

    #Parses XML and extracts data points
    root = ET.fromstring(response.content)
    rows = []

    for obs in root.findall('.//omso:PointTimeSeriesObservation', ns):
        observed = obs.find('om:observedProperty', ns)
        href = observed.get('{http://www.w3.org/1999/xlink}href', '') if observed is not None else ''

        for mts in obs.findall('.//wml2:MeasurementTimeseries', ns):
            series_id = mts.get('{http://www.opengis.net/gml/3.2}id', '').lower()

            #Identifies parameters from serie-id or href
            param_key = None
            for p in series_map:
                if p in series_id or (href and p in href):
                    param_key = p
                    break
            if param_key is None:
                if debug:
                    print("Skipping series (unknown param):", series_id, href)
                continue
            col = series_map[param_key]

            #Extracts time and value for every data point
            for point in mts.findall('.//wml2:point', ns):
                tvp = point.find('wml2:MeasurementTVP', ns)
                if tvp is None:
                    continue
                t = tvp.find('wml2:time', ns)
                v = tvp.find('wml2:value', ns)
                if t is None or v is None:
                    continue
                try:
                    val = float(v.text)
                except (TypeError, ValueError):
                    continue
                rows.append({"timestamp": t.text, col: val})

    if not rows:
        if debug:
            print("No measurement rows parsed; inspect response XML.")
        return pd.DataFrame()
    
    #Creates DataFrame from extracted rows
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.pivot_table(index='timestamp', aggfunc='first').reset_index()

    #Ensures that all columns exist
    for col in ("temperature", "wind_speed", "wind_direction", "humidity", "pressure"):
        if col not in df.columns:
            df[col] = np.nan

    #Converts columns to numeric values and fills NaN
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(0.0)
    df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce').fillna(0.0)
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce').fillna(0.0)
    df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce').fillna(1013.0)

    #Sorts time and calculates cyclic representation of wind direction
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['winddir_sin'] = np.sin(np.radians(df['wind_direction']))
    df['winddir_cos'] = np.cos(np.radians(df['wind_direction']))

    return df
