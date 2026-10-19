"""Bounded local plasticity with eligibility, STDP and learned modulation."""
import torch
from torch import nn


class Plasticity(nn.Module):
    def __init__(self, pre, post, decay=0.9, rate=0.01, limit=1.0):
        super().__init__()
        if pre < 1 or post < 1 or not 0 <= decay < 1 or rate <= 0 or limit <= 0:
            raise ValueError("Invalid plasticity configuration")
        self.pre, self.post, self.decay, self.rate, self.limit = pre, post, decay, rate, limit
        self.modulator = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1), nn.Tanh())
        self.register_buffer("trace", torch.zeros(post, pre))
        self.register_buffer("fast", torch.zeros(post, pre))
        self.register_buffer("pre_trace", torch.zeros(pre))
        self.register_buffer("post_trace", torch.zeros(post))

    def update(self, pre, post, reward=1.0, mode="eligibility"):
        if pre.ndim != 2 or post.ndim != 2 or pre.shape != (post.shape[0], self.pre) or post.shape[1] != self.post:
            raise ValueError("Plasticity batch dimensions differ")
        if mode not in {"hebbian", "stdp", "eligibility", "learned"}:
            raise ValueError("Unknown plasticity rule")
        if not torch.isfinite(pre).all() or not torch.isfinite(post).all() or not torch.isfinite(torch.tensor(reward)):
            raise ValueError("Nonfinite plasticity signal")
        correlation = post.T @ pre / pre.shape[0]
        if mode == "stdp":
            correlation = torch.outer(post.mean(0), self.pre_trace) - torch.outer(self.post_trace, pre.mean(0))
        eligibility = self.decay * self.trace + correlation
        signal = torch.as_tensor(reward, dtype=pre.dtype, device=pre.device)
        if mode == "learned":
            features = torch.stack((pre.mean(), post.mean(), signal))
            signal = self.modulator(features).squeeze()
        delta = correlation if mode in {"hebbian", "stdp"} else eligibility
        proposal = (self.fast + self.rate * signal * delta).clamp(-self.limit, self.limit)
        with torch.no_grad():
            self.trace.copy_(eligibility.detach())
            self.fast.copy_(proposal.detach())
            self.pre_trace.mul_(self.decay).add_(pre.mean(0).detach(), alpha=1 - self.decay)
            self.post_trace.mul_(self.decay).add_(post.mean(0).detach(), alpha=1 - self.decay)
        return proposal

    def forward(self, value):
        return value @ self.fast.T

    def reset(self):
        for value in (self.trace, self.fast, self.pre_trace, self.post_trace):
            value.zero_()
