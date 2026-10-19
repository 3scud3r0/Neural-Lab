"""Small partially observed resource/tool/construction world, PettingZoo Parallel API."""
from dataclasses import dataclass

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv


@dataclass(frozen=True)
class WorldConfig:
    size: int = 7
    horizon: int = 60
    sight: int = 2
    resources: int = 10
    build_cost: int = 3
    vocabulary: int = 8

    def __post_init__(self):
        if self.size < 3 or min(self.horizon, self.sight, self.resources, self.build_cost, self.vocabulary) < 1:
            raise ValueError("Invalid world configuration")


class ConstructionWorld(ParallelEnv):
    metadata = {"name": "ouroboros_construction_v3", "render_modes": []}
    possible_agents = ["agent_0", "agent_1"]

    def __init__(self, config=WorldConfig()):
        self.config = config
        self.agents = []
        self._action = spaces.MultiDiscrete([7, config.vocabulary])
        self._observation = spaces.Box(-1, 1, shape=(16,), dtype=np.float32)

    def action_space(self, agent):
        return self._action

    def observation_space(self, agent):
        return self._observation

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self.positions = {a: self.rng.integers(0, self.config.size, 2) for a in self.agents}
        self.resources = np.zeros((self.config.size, self.config.size), dtype=np.int64)
        for _ in range(self.config.resources):
            self.resources[tuple(self.rng.integers(0, self.config.size, 2))] += 1
        self.inventory = {a: 0 for a in self.agents}
        self.tools = {a: False for a in self.agents}
        self.energy = {a: 1.0 for a in self.agents}
        self.messages = {a: 0 for a in self.agents}
        self.clues = {a: int(self.rng.integers(1, self.config.vocabulary + 1)) for a in self.agents}
        self.structure, self.time = 0, 0
        return {a: self.observe(a) for a in self.agents}, {a: {} for a in self.agents}

    def observe(self, agent):
        position = self.positions[agent]
        other = next(a for a in self.possible_agents if a != agent)
        delta = self.positions[other] - position
        visible = np.max(np.abs(delta)) <= self.config.sight
        local = []
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = position + (dx, dy)
            valid = np.all(cell >= 0) and np.all(cell < self.config.size)
            local.append(min(self.resources[tuple(cell)], 4) / 4 if valid else -1)
        result = [*(position / (self.config.size - 1)), *local,
                  self.inventory[agent] / max(self.config.resources, 1), float(self.tools[agent]),
                  self.energy[agent], min(self.structure / 4, 1), self.time / self.config.horizon,
                  self.messages[other] / self.config.vocabulary if visible else 0,
                  self.clues[agent] / self.config.vocabulary,
                  *(delta / self.config.size if visible else np.zeros(2))]
        return np.clip(np.asarray(result, dtype=np.float32), -1, 1)

    def step(self, actions):
        if set(actions) != set(self.agents):
            raise ValueError("Exactly one action per active agent required")
        if any(not self._action.contains(np.asarray(action)) for action in actions.values()):
            raise ValueError("Invalid action")
        active = self.agents.copy()
        rewards = {a: 0.0 for a in active}
        # Seeded random execution order avoids a permanent first-agent advantage.
        for agent in self.rng.permutation(active):
            action, message = actions[agent]
            self.messages[agent] = int(message)
            self.energy[agent] = max(0, self.energy[agent] - 0.005 - 0.001 * bool(message))
            if self.energy[agent] <= 0:
                continue
            if action < 4:
                displacement = ((1, 0), (-1, 0), (0, 1), (0, -1))[action]
                self.positions[agent] = np.clip(self.positions[agent] + displacement, 0, self.config.size - 1)
            elif action == 4:
                location = tuple(self.positions[agent])
                amount = min(self.resources[location], 2 if self.tools[agent] else 1)
                self.resources[location] -= amount
                self.inventory[agent] += int(amount)
                rewards[agent] += 0.05 * amount
            elif action == 5 and not self.tools[agent] and self.inventory[agent] >= 2:
                self.inventory[agent] -= 2
                self.tools[agent] = True
            elif action == 6 and self.inventory[agent] >= self.config.build_cost:
                self.inventory[agent] -= self.config.build_cost
                self.structure += 1
                for other in active:
                    rewards[other] += 1.0
        self.time += 1
        terminal = self.structure >= 4 or all(self.energy[a] <= 0 for a in active)
        truncated = self.time >= self.config.horizon and not terminal
        observations = {a: self.observe(a) for a in active}
        terminations, truncations = {a: terminal for a in active}, {a: truncated for a in active}
        infos = {a: {"structure": self.structure} for a in active}
        if terminal or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos
