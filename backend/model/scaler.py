#Creates scaler for model for normalization of models input data
#Normalization speeds up the models training and causes the model to make more accurate predictions
from dataclasses import dataclass
import torch

@dataclass
class Scaler:
    #Statistics that are used for normalization of variables
    temp_mean: float
    temp_std: float
    wind_mean: float
    wind_std: float
    temp_diff_std: float
    pressure_mean: float
    pressure_std: float

    def save(self, path: str):
        #Saves scaler as a Pytorch file
        torch.save(
            {
                "temp_mean": float(self.temp_mean),
                "temp_std": float(self.temp_std),
                "wind_mean": float(self.wind_mean),
                "wind_std": float(self.wind_std),
                "temp_diff_std": float(self.temp_diff_std),
                "pressure_mean": float(self.pressure_mean),
                "pressure_std": float(self.pressure_std),
            },
            path
        )

    @staticmethod
    def load(path: str):
        #Reads saved scaler from file and returns a Scaler-object that can be used
        data = torch.load(path, map_location="cpu")
        return Scaler(**data)
