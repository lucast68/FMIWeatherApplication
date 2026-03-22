#Client module for communication with Flask
import requests

API_URL = 'http://127.0.0.1:5000'

#Fetches weather data from Flask server
def get_weather(place="Helsinki"):
    try:
        response = requests.get(f"{API_URL}/weather", params={"place": place})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Issues with fetching weather data: {e}") from e

#Fetches temperature predictions from Flask server
def get_prediction(place="Helsinki"):
    try:
        response = requests.get(f"{API_URL}/predict", params={"place": place})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Issues with fetching prediction data: {e}") from e
