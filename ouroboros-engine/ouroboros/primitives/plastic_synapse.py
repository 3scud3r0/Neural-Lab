import torch
from torch import nn
from ouroboros.core.meta_plasticity import hebbian_update


class PlasticSynapse(nn.Module):
    def __init__(self, size, population, learning_rate=0.025, decay=0.03, clip=1.0):
        super().__init__()
        self.fixed = nn.Linear(size, size, bias=False)
        self.register_buffer("trace", torch.zeros(population, size, size))
        self.rate, self.decay, self.limit = learning_rate, decay, clip

    def forward(self, x, modulation):
        # clone separa memória mutável dos tensores usados pelo autograd.
        y = torch.tanh(self.fixed(x) + torch.bmm(self.trace.detach().clone(), x.unsqueeze(-1)).squeeze(-1))
        hebbian_update(self.trace, x.detach(), y.detach(), self.rate, self.decay, self.limit, modulation)
        return y
