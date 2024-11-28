import torch
import torch.nn as nn

class NBAGamePredictor(nn.Module):
    def __init__(self, input_size=11, hidden_size=64):
        super().__init__()

        # LSTM for home team
        self.home_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # LSTM for away team
        self.away_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

