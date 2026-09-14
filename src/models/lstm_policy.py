"""
lstm_policy.py
Policy Architecture: Recurrent LSTM Encoder + Deep MLP Decision Head.
Parameter count: exactly 216,321 trainable parameters.
"""
import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    """
    Recurrent policy network for residential EV charging energy management.
    Processes historical 24-hour time series of electricity price, rooftop PV,
    and residential baseload, fused with current time-of-day and battery SoC.
    
    Architecture:
      - LSTM Encoder: input_size=3 (price, pv, load), hidden_size=64, layers=1
      - Fusion layer: concatenation of LSTM hidden state (64) + time-of-day (1) + SoC (1) = 66
      - MLP Head: Linear(66 -> 512) -> ReLU -> Linear(512 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 1)
      - Activation: Hardtanh(min_val=e_min, max_val=e_max)
    """
    def __init__(self, state_dim: int = 74, action_dim: int = 1,
                 e_min: float = -7.0, e_max: float = 7.0,
                 hidden1: int = 512, hidden2: int = 256, hidden3: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.e_min = e_min
        self.e_max = e_max
        
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(64 + 2, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)
        self.linear4 = nn.Linear(hidden3, action_dim)
        self.tanh = nn.Hardtanh(min_val=e_min, max_val=e_max)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: Tensor of shape (batch_size, 74)
              - indices 0..23:  price series (24 steps)
              - indices 24..47: pv series (24 steps)
              - indices 48..71: load series (24 steps)
              - index 72:       time of day (normalized)
              - index 73:       current battery SoC (normalized)
        Returns:
            Continuous EV power dispatch action (batch_size, 1) in [-7.0, 7.0] kW
        """
        price = states[:, 0:24].unsqueeze(-1)
        pv = states[:, 24:48].unsqueeze(-1)
        load = states[:, 48:72].unsqueeze(-1)
        series_3d = torch.cat([price, pv, load], dim=-1)

        _, (h_n, _) = self.lstm(series_3d)
        lstm_feat = h_n[-1]

        scalars = states[:, 72:74]
        fused = torch.cat([lstm_feat, scalars], dim=1)

        x = torch.relu(self.linear1(fused))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        action = self.tanh(self.linear4(x))
        return action
