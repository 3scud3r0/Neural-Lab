import torch
from torch.nn import functional as F


def predictive_loss(prediction, observation, activations):
    """Erro quadrático e regularização de atividade; não ELBO variacional."""
    accuracy_error = F.mse_loss(prediction, observation)
    complexity = activations.square().mean()
    return accuracy_error + 0.001 * complexity


def activation_entropy(activations):
    probabilities = torch.softmax(activations.detach(), dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
