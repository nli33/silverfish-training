import torch
import torch.nn as nn


NUM_INPUTS = 64 * 12 # 768
L1 = 1024


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(NUM_INPUTS, L1)
        self.output = nn.Linear(2 * L1, 1)


    def forward(self, x_white: torch.Tensor, x_black: torch.Tensor, white_to_move: torch.Tensor):
        h_white = torch.relu(self.input(x_white))
        h_black = torch.relu(self.input(x_black))
        h = torch.where(
            white_to_move.unsqueeze(1),
            torch.cat([h_white, h_black], dim=1),
            torch.cat([h_black, h_white], dim=1)
        )
        return self.output(h).squeeze(1)
