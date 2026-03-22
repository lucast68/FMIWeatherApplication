
#Creates Feedforward Neural Network with three linear layers
import torch.nn as nn

#Amount of indata features that the model receives
INPUT_DIM = 23


#Neural network for temperature predictions
#Model receives weather related features as indata and predicts the temperature one hour forward in time.
class WeatherModel(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()

        #Feedforward Neural Network (FNN)
        self.net = nn.Sequential(
            #First hidden layer
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            #Second hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            #Output layer (regression, 1 value)
            nn.Linear(32, 1)
        )
    
    #Passes forward through the network
    def forward(self, x):
        return self.net(x)
