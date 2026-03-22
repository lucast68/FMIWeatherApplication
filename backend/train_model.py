#Imports necessary packages for PyTorch and data processing
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from model.model import WeatherModel
from model.scaler import Scaler
from utils.preprocess import prepare_features
from fmi_fetch import fetch_fmi_weather_data

#Trains weather model
def train_model():
    #Fetches weather data from measurement station in Kaisaniemi
    df = fetch_fmi_weather_data("Kaisaniemi", debug=True)
    if df is None:
        raise RuntimeError("fetch_fmi_weather_data returned None. Check FMI request/response (enable debug).")
    if df.empty:
        raise RuntimeError("fetch_fmi_weather_data returned an empty DataFrame. Check request params (place/fmisid/starttime/endtime).")

    #Prepares training data: features and goal variables
    df = df.reset_index(drop=True)
    df_features = df.iloc[:-1].reset_index(drop=True)
    y_series = df['temperature'].shift(-1).dropna().reset_index(drop=True)

    #Creates scaler for normalization of input data and goal variable
    scaler = Scaler(
        temp_mean=df["temperature"].mean(),
        temp_std=df["temperature"].std(),
        wind_mean=df["wind_speed"].mean(),
        wind_std=df["wind_speed"].std(),
        temp_diff_std=df["temperature"].diff().std(),
        pressure_mean=df["pressure"].mean(),
        pressure_std=df["pressure"].std()
    )

    #Extracts features and normalizes goal variables
    X = prepare_features(df_features, scaler)
    y = (y_series.values - scaler.temp_mean) / scaler.temp_std
    y = torch.from_numpy(y).float().unsqueeze(1)
    
    #Creates DataLoader for batch training
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    #Intiates model, optimizer and loss function
    model = WeatherModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epochs = 100

    #Training loop
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Shows loss for monitoring
            if epoch % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    #Saves trained model and scaler
    torch.save(model.state_dict(), 'trained_model/weather_model.pth')
    print("Model sparad som 'trained_model/weather_model.pth'")

    scaler.save("trained_model/scaler.pth")
    print("Scaler sparad som 'trained_model/scaler.pth'")

if __name__ == "__main__":
    train_model()
