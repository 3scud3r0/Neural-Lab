import torch
from torch import nn
from torch.distributions import Normal
from ouroboros.primitives import LiquidCell, PlasticSynapse, SpikingGate, TTTLayer


class DynamicBrain(nn.Module):
    """Pesos lentos compartilhados, estados e memórias rápidos individuais."""

    def __init__(self, input_size, hidden_size, signal_size, population, primitives):
        super().__init__()
        self.liquid = LiquidCell(input_size, hidden_size, **primitives["liquid"])
        self.spiking = SpikingGate(**primitives["spiking"])
        self.plastic = PlasticSynapse(hidden_size, population, **primitives["plastic"])
        self.ttt = TTTLayer(hidden_size, population, **primitives["ttt"])
        self.motor = nn.Linear(hidden_size, 2)
        self.voice = nn.Linear(hidden_size, signal_size)
        self.predictor = nn.Linear(hidden_size, input_size)
        self.register_buffer("hidden", torch.zeros(population, hidden_size))

    def forward(self, observation, dt, modulation):
        liquid = self.liquid(observation, self.hidden.detach().clone(), dt)
        gated, spikes = self.spiking(liquid)
        memory = self.ttt(self.plastic(gated, modulation))
        # Via residual mantém informações quando poucos neurônios disparam.
        activity = torch.tanh(liquid + memory)
        distribution = Normal(self.motor(activity).tanh(), 0.25)
        raw_action = distribution.sample()
        action = raw_action.clamp(-1, 1)
        voice_distribution = Normal(self.voice(activity).tanh(), 0.15)
        raw_signal = voice_distribution.sample()
        with torch.no_grad():
            self.hidden.copy_(liquid.detach())
        return {"action": action, "log_prob": distribution.log_prob(raw_action).sum(-1),
                "signal_log_prob": voice_distribution.log_prob(raw_signal).sum(-1),
                "signal": raw_signal.clamp(-1, 1), "prediction": self.predictor(activity),
                "activity": activity, "spikes": spikes}

    @torch.no_grad()
    def reset_agents(self, mask):
        self.hidden[mask] = 0
        self.plastic.trace[mask] = 0
        self.ttt.fast[mask] = 0
        self.ttt.previous[mask] = 0
