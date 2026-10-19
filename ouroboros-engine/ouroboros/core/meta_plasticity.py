import torch


@torch.no_grad()
def hebbian_update(trace, pre, post, rate, decay, limit, modulation):
    """Atualização por lote [agentes, saída, entrada], CPU ou CUDA nativo."""
    outer = post.unsqueeze(-1) * pre.unsqueeze(-2)
    trace.mul_(1 - decay).add_(rate * modulation[:, None, None] * outer)
    trace.clamp_(-limit, limit)
