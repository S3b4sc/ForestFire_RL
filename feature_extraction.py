import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# Define the custor CC Feature Extractor
class CustomCNNFeaturExtractor(BaseFeaturesExtractor):
    """
    """
    def __init__(self, observation_space:gym.spaces.Box, features_dim:int = 512, normalized_image: bool = False) -> None:
        # Call parent constructor
        super(CustomCNNFeaturExtractor,self).__init__(observation_space, features_dim)
        
        # imput dimensions from the observation space (environment)
        #print(observation_space.shape)
        input_channels = observation_space.shape[0]
        grid_height = observation_space.shape[1]
        grid_width = observation_space.shape[2]
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels,32, kernel_size=3,stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=2, stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=2, stride=1,padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Size of the calculated flattened output
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, grid_height, grid_width)
            cnn_output_dim = self.cnn(dummy_input).shape[1]
            
        # Use a fully connected layers (Linear)
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )
        
        # Update feature dimension
    
    def forward(self, observations:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN and FC layers
        """
        cnn_output = self.cnn(observations)
        return self.fc(cnn_output)
    
