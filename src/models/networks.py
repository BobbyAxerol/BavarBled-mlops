import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerViewGenerator(nn.Module):
    def __init__(self, num_assets=29, lookback=15, num_features=12, d_model=128, nhead=2, num_layers=4):
        super(TransformerViewGenerator, self).__init__()
        # Input shape: (Batch, Assets, Lookback, Features) -> Flatten features/lookback or process sequentially
        self.input_projection = nn.Linear(lookback * num_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1) # Outputs a scalar view value per asset

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, 29, 15, 12)
        B, N, W, F_dim = state_tensor.shape
        flat_state = state_tensor.view(B, N, W * F_dim)
        
        x = self.input_projection(flat_state) # Shape: (B, N, d_model)
        # Note: Positional Encoding omitted deliberately to focus entirely on asset cross-correlations
        x = self.transformer(x) # Shape: (B, N, d_model)
        Q = self.output_layer(x).squeeze(-1) # Shape: (B, N)
        return Q

class CNNRiskNetwork(nn.Module):
    def __init__(self, num_assets=29, lookback=15, num_features=12, hidden_size=512):
        super(CNNRiskNetwork, self).__init__()
        # Permute state tensor for spatial 2D convolution over (Lookback, Features)
        self.conv1 = nn.Conv2d(in_channels=num_assets, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, hidden_size, kernel_size=(3, 3), padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor: (B, 29, 15, 12)
        x = F.relu(self.conv1(state_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1) # Shape: (B, hidden_size)
        
        x = F.relu(self.fc1(x))
        delta = torch.sigmoid(self.fc2(x)) * 5.0 + 0.1 # Bounded output scalar within range [0.1, 5.1]
        return delta