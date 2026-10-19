import torch
from torch import nn


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + ctx.slope * x.abs()).square(), None


class SpikingGate(nn.Module):
    """Gate binário com gradiente substituto; operações continuam densas."""

    def __init__(self, threshold=0.35, surrogate_slope=5.0):
        super().__init__()
        self.threshold, self.slope = threshold, surrogate_slope

    def forward(self, x):
        spikes = _Spike.apply(x.abs() - self.threshold, self.slope)
        return x * spikes, spikes
