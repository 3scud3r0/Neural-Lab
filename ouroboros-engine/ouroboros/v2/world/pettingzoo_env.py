"""Adaptador real da API ParallelEnv de PettingZoo para a arena em tensores."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo import ParallelEnv
import torch

from .arena import ArenaConfig, CooperativeArena
from ..communication.game import GameConfig


class CooperativeParallelEnv(ParallelEnv):
    metadata = {"name": "ouroboros_cooperative_v2", "render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, config=None, game=None, render_mode=None):
        super().__init__()
        self.config = replace(config or ArenaConfig(), batch=1)
        self.game = game or GameConfig()
        self.arena = CooperativeArena(self.config, self.game)
        self.possible_agents = ["scout", "worker"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode
        self._feature_size = self.arena.observation()["features"].shape[-1]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Dict({"thrust": spaces.Box(-1, 1, (2,), dtype=np.float32),
                            "grip": spaces.Discrete(2),
                            "choice": spaces.Discrete(self.game.candidates), "commit": spaces.Discrete(2),
                            "message": spaces.MultiDiscrete([self.game.vocabulary] * self.game.message_length)})

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict({"features": spaces.Box(-2, 2, (self._feature_size,), dtype=np.float32),
                            "message": spaces.MultiDiscrete([self.game.vocabulary] * self.game.message_length),
                            "received": spaces.Discrete(2)})

    def _observations(self):
        observation = self.arena.observation()
        return {name: {"features": observation["features"][0, i].cpu().numpy().astype(np.float32),
                       "message": observation["messages"][0, i].cpu().numpy(),
                       "received": int(observation["received"][0, i])}
                for i, name in enumerate(self.possible_agents)}

    def reset(self, seed=None, options=None):
        self.arena.reset(seed)
        self.agents = self.possible_agents.copy()
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents): raise ValueError("Ações precisam cobrir ambos os agentes ativos")
        thrust = torch.tensor(np.stack([actions[name]["thrust"] for name in self.possible_agents]), dtype=torch.float32)[None]
        grip = torch.tensor([[actions[name]["grip"] for name in self.possible_agents]])
        messages = torch.tensor(np.stack([actions[name]["message"] for name in self.possible_agents]), dtype=torch.long)[None]
        choices = torch.tensor([[int(actions[name]["choice"]) if actions[name]["commit"] else -1
                                 for name in self.possible_agents]], dtype=torch.long)
        _, reward, done, info = self.arena.step(thrust, grip, messages, choices)
        truncated = self.arena.steps >= self.config.horizon and not bool(self.arena.success[0])
        terminated = bool(done[0]) and not truncated
        observations = self._observations()
        rewards = {agent: float(reward[0]) for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"harvest": bool(info["harvest"][0])} for agent in self.agents}
        if bool(done[0]): self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self):
        return torch.cat((self.arena.observation()["features"].flatten(), self.arena.target.float())).numpy()

    def render(self):
        canvas = np.zeros((480, 480, 3), dtype=np.uint8)
        canvas[:] = (14, 23, 37)
        def disc(position, color, radius):
            x, y = (position / self.config.size * 479).long().clamp(0, 479).tolist()
            yy, xx = np.ogrid[:480, :480]
            canvas[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = color
        for p in self.arena.objects[0]: disc(p, (240, 185, 94), 10)
        for p in self.arena.obstacles[0]: disc(p, (183, 80, 100), 12)
        for p, color in zip(self.arena.positions[0], [(91, 223, 182), (119, 160, 255)]): disc(p, color, 6)
        return canvas

    def close(self):
        self.agents = []
