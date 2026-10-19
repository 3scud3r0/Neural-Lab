import torch
from torch import nn
from torch.nn import functional as F


class TTTLayer(nn.Module):
    """Memória rápida: SGD analítico de um preditor linear por organismo.

    Aprende a prever a ativação seguinte; implementação inspirada em TTT,
    sem equivalência aos modelos publicados de larga escala.
    """

    def __init__(self, size, population, learning_rate=0.03, decay=0.01):
        super().__init__()
        self.register_buffer("fast", torch.zeros(population, size, size))
        self.register_buffer("previous", torch.zeros(population, size))
        self.rate, self.decay = learning_rate, decay

    def forward(self, x):
        with torch.no_grad():
            previous = F.normalize(self.previous, dim=-1)
            prediction = torch.bmm(self.fast, previous.unsqueeze(-1)).squeeze(-1)
            error = x.detach() - prediction
            self.fast.mul_(1 - self.decay).add_(self.rate * error.unsqueeze(-1) * previous.unsqueeze(-2))
            self.fast.clamp_(-2, 2)
            self.previous.copy_(x.detach())
        readout = torch.bmm(self.fast.detach().clone(), F.normalize(x, dim=-1).unsqueeze(-1)).squeeze(-1)
        return torch.tanh(x + readout)
