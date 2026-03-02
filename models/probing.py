import torch
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_


class ProbingMLP(nn.Module):
    """Probing MLP that decodes digit predictions from concatenated residual streams.

    Input:  concat(z_H, z_L, input_emb) of shape [batch, seq, 3 * hidden_size]
    Output: logits over 9 classes (digits 1-9) of shape [batch, seq, 9]
    """

    def __init__(self, hidden_size: int = 512, hidden_mult: int = 4, num_classes: int = 9):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_mult = hidden_mult
        self.num_classes = num_classes

        d_in = 3 * hidden_size
        d_hidden = hidden_mult * d_in

        # Truncated LeCun normal init, following codebase conventions
        self.fc1_weight = nn.Parameter(
            trunc_normal_init_(torch.empty(d_hidden, d_in), std=1.0 / (d_in ** 0.5))
        )
        self.fc1_bias = nn.Parameter(torch.zeros(d_hidden))

        self.fc2_weight = nn.Parameter(
            trunc_normal_init_(torch.empty(num_classes, d_hidden), std=1.0 / (d_hidden ** 0.5))
        )
        self.fc2_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_in]
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = F.relu(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        # x: [batch, seq, num_classes]
        return x
