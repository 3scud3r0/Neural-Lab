"""Independent recurrent agents, inheritance, social distillation and free symbols."""
from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn

from .architecture import HeterogeneousBrain, Region
from .memory import EpisodicMemory


@dataclass
class Individual:
    identity: int
    brain: HeterogeneousBrain
    memory: EpisodicMemory
    generation: int = 0
    reward: float = 0.0
    parent: int | None = None


class Population:
    def __init__(self, count, expression, hidden=12, vocabulary=8):
        if count < 2:
            raise ValueError("A society needs at least two individuals")
        self.vocabulary = vocabulary
        self.members = [Individual(i, HeterogeneousBrain(16, 7 + vocabulary,
                        [Region("cortex", hidden, expression)]), EpisodicMemory()) for i in range(count)]
        self.next_identity = count
        self.optimizers = {member.identity: torch.optim.Adam(member.brain.parameters(), lr=0.003)
                           for member in self.members}

    def reproduce(self, parent_index, mutation=0.02):
        parent = self.members[parent_index]
        brain = copy.deepcopy(parent.brain)
        with torch.no_grad():
            for parameter in brain.parameters():
                parameter.add_(torch.randn_like(parameter) * mutation)
        child = Individual(self.next_identity, brain, EpisodicMemory(), parent.generation + 1, parent=parent.identity)
        self.next_identity += 1
        victim = min((i for i in range(len(self.members)) if i != parent_index),
                     key=lambda i: self.members[i].reward)
        self.optimizers.pop(self.members[victim].identity)
        self.members[victim] = child
        self.optimizers[child.identity] = torch.optim.Adam(child.brain.parameters(), lr=0.003)
        return child

    def distill(self, teacher_index, student_index, observations, steps=5):
        if teacher_index == student_index:
            raise ValueError("Distinct teacher and student required")
        teacher, student = self.members[teacher_index].brain, self.members[student_index].brain
        with torch.no_grad():
            targets, _ = teacher(observations)
            probabilities = targets.softmax(-1)
        optimizer = self.optimizers[self.members[student_index].identity]
        history = []
        for _ in range(steps):
            prediction, _ = student(observations)
            loss = -(probabilities * prediction.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def episode(self, environment, seed, train=False, intervention="intact"):
        if intervention not in {"intact", "muted", "shuffled"}:
            raise ValueError("Unknown communication intervention")
        observations, _ = environment.reset(seed=seed)
        mapping = dict(zip(environment.possible_agents, self.members))
        states, logs, rewards = {}, {a: [] for a in mapping}, {a: [] for a in mapping}
        optimizers = {a: self.optimizers[member.identity] for a, member in mapping.items()}
        intervention_rng = np.random.default_rng(seed + 902)
        totals = {a: 0.0 for a in mapping}
        while environment.agents:
            actions = {}
            for agent in environment.agents:
                member = mapping[agent]
                observation = torch.as_tensor(observations[agent]).unsqueeze(0)
                logits, state = member.brain(observation, states.get(agent))
                states[agent] = {name: value.detach() for name, value in state.items()}
                physical = torch.distributions.Categorical(logits=logits[0, :7])
                symbolic = torch.distributions.Categorical(logits=logits[0, 7:])
                action = physical.sample() if train else physical.logits.argmax()
                message = symbolic.sample() if train else symbolic.logits.argmax()
                logs[agent].append(physical.log_prob(action) + symbolic.log_prob(message))
                if intervention == "muted": message = message * 0
                actions[agent] = np.asarray([int(action), int(message)])
                member.memory.add(observation[0], {"action": int(action), "message": int(message)})
            if intervention == "shuffled":
                symbols = intervention_rng.permutation([action[1] for action in actions.values()])
                for action, symbol in zip(actions.values(), symbols):
                    action[1] = symbol
            observations, received, _, _, _ = environment.step(actions)
            for agent in mapping:
                rewards[agent].append(received[agent])
                totals[agent] += received[agent]
        if train:
            for agent, member in mapping.items():
                returns, accumulated = [], 0.0
                for reward in reversed(rewards[agent]):
                    accumulated = reward + 0.97 * accumulated
                    returns.append(accumulated)
                returns = torch.tensor(list(reversed(returns)))
                loss = -(torch.stack(logs[agent]) * returns).mean()
                optimizers[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(member.brain.parameters(), 1.0)
                optimizers[agent].step()
                member.reward += totals[agent]
        return {"rewards": totals, "structure": environment.structure, "steps": environment.time,
                "intervention": intervention}


def train_free_language(directory, steps=300, seed=42):
    # Reuse the already tested free GRU game; no attribute-specific symbol slots.
    from ..v2.communication import CommunicationGame, GameConfig, train_communication
    from .experiments import isolated_seed
    with isolated_seed(seed):
        return train_communication(CommunicationGame(GameConfig(protocol="free")), directory,
                                   steps=steps, seed=seed)
