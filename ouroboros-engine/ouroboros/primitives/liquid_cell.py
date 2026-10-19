import torch
from torch import nn


class LiquidCell(nn.Module):
    """Relaxação exponencial com tau condicionado ao estado (não CfC canônica)."""

    def __init__(self, input_size, hidden_size, tau_min=0.1, tau_max=2.0):
        super().__init__()
        self.target = nn.Linear(input_size + hidden_size, hidden_size)
        self.tau = nn.Linear(input_size + hidden_size, hidden_size)
        self.tau_min, self.tau_max = tau_min, tau_max

    def forward(self, x, hidden, dt):
        if dt <= 0:
            raise ValueError("dt precisa ser positivo")
        joined = torch.cat((x, hidden), dim=-1)
        tau = self.tau_min + (self.tau_max - self.tau_min) * self.tau(joined).sigmoid()
        alpha = -torch.expm1(-dt / tau)
        return hidden + alpha * (self.target(joined).tanh() - hidden)
