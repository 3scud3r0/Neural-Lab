"""Liga símbolos discretos, CfC upstream, morfogênese e cooperação física."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ..common import Journal, atomic_json, wilson_interval
from ..communication.game import CommunicationGame
from ..evolution.morphology import MorphogenicBrain, MorphologyConfig
from .arena import ArenaConfig, CooperativeArena


@torch.no_grad()
def sensory_input(arena, language, channel="intact", generator=None):
    """Apenas o scout usa target. O worker escolhe através do receptor treinado."""
    if channel not in {"intact", "muted", "shuffled"}: raise ValueError("Canal desconhecido")
    sent, _ = language.sender(arena.target_features(), discrete=True)
    transmitted = sent.argmax(-1)
    if channel == "muted":
        message = torch.zeros_like(sent)
    elif channel == "shuffled":
        order = torch.randperm(arena.config.batch, device=arena.device, generator=generator)
        message = sent[order]
    else:
        message = sent
    # O canal efetivo é discrete e sujeito ao alcance do rádio da arena.
    _, received = arena.receive()
    message = message * received[:, None, None]
    selection = language.receiver(message, arena.candidate_features()).argmax(-1)
    # A escolha inicial é uma chave irreversível; seguir o scout depois não corrige uma chave errada.
    selection = torch.where(arena.claims[:, 1] >= 0, arena.claims[:, 1], selection)
    choices = torch.stack((arena.target, selection), -1)
    destinations = arena.objects.gather(1, choices[..., None].expand(-1, -1, 2))
    direction = (destinations - arena.positions) / arena.config.size
    partner = (arena.positions.flip(1) - arena.positions) / arena.config.size
    role = torch.tensor([0., 1.], device=arena.device)[None, :, None].expand(arena.config.batch, -1, -1)
    if arena.config.obstacles:
        distances = torch.cdist(arena.positions, arena.obstacles)
        nearest = distances.argmin(-1)
        obstacle = arena.obstacles.gather(1, nearest[..., None].expand(-1, -1, 2))
        obstacle = (obstacle - arena.positions) / arena.config.size
    else:
        obstacle = torch.ones_like(direction)
    observation = torch.cat((direction, arena.velocity / arena.config.speed_limit,
                             arena.energy[..., None], role, partner,
                             torch.full_like(role, arena.daylight), obstacle,
                             received[:, None, None].expand(-1, 2, -1).float()), -1)
    messages = torch.zeros_like(arena.messages)
    messages[:, 0] = transmitted
    messages[:, 1] = transmitted  # Confirmação de recebimento, sem nova informação privada.
    return observation.reshape(-1, 12), messages, selection


def navigation_teacher(observation, config):
    """Currículo explícito de navegação; só acessa as mesmas entradas da política."""
    delta = observation[:, :2] * config.size
    velocity = observation[:, 2:4] * config.speed_limit
    obstacle = observation[:, 9:11] * config.size
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    desired_velocity = delta / distance.clamp_min(0.6) * torch.minimum(distance, torch.ones_like(distance))
    thrust = (desired_velocity - velocity) * 1.4
    obstacle_distance = torch.linalg.vector_norm(obstacle, dim=-1, keepdim=True)
    proximity = (0.65 - obstacle_distance).clamp_min(0) / 0.65
    thrust -= obstacle / obstacle_distance.clamp_min(0.1) * proximity * 1.5
    grip = distance.squeeze(-1) < config.radius
    return thrust.clamp(-1, 1), grip.float()


@torch.no_grad()
def evaluate_world(brain, language, config=None, episodes=128, seed=1234, channel="intact", solo=False):
    config = config or ArenaConfig()
    config = ArenaConfig(**dict(asdict(config), batch=episodes))
    device = next(brain.parameters()).device
    arena = CooperativeArena(config, language.config, device)
    arena.reset(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    state = None
    was_training = brain.training
    brain.eval()
    total_reward = torch.zeros(episodes, device=device)
    for _ in range(config.horizon):
        observation, messages, selection = sensory_input(arena, language, channel, generator)
        output = brain(observation, state, config.dt)
        state = output["state"]
        grip = (output["grip_logit"].sigmoid() > 0.5).reshape(episodes, 2)
        if solo: grip[:, 1] = False
        choices = torch.stack((arena.target, selection), -1)
        _, reward, _, _ = arena.step(output["thrust"].reshape(episodes, 2, 2), grip, messages, choices)
        total_reward += reward
        if bool(arena.done.all()): break
    successes = int(arena.success.sum())
    brain.train(was_training)
    return {"channel": channel, "solo": solo, "successes": successes, "episodes": episodes,
            "success_rate": successes / episodes, "ci95": wilson_interval(successes, episodes),
            "mean_reward": float(total_reward.mean()), "snapshot": arena.snapshot()}


def train_world(language, directory: Path, episodes=24, config=None, morphology=None, seed=42):
    if episodes < 1: raise ValueError("Episódios devem ser positivos")
    config, morphology = config or ArenaConfig(), morphology or MorphologyConfig()
    if morphology.inputs != 12: raise ValueError("O controlador físico utiliza 12 entradas")
    device = next(language.parameters()).device
    brain = MorphogenicBrain(morphology).to(device)
    arena = CooperativeArena(config, language.config, device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.002)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory)
    language.eval()
    history = []
    for episode in range(episodes):
        brain.train()
        arena.reset(seed + episode)
        state = None
        losses, rewards = [], []
        for step in range(config.horizon):
            observation, messages, selection = sensory_input(arena, language)
            output = brain(observation, state, config.dt)
            state = output["state"]
            teacher_thrust, teacher_grip = navigation_teacher(observation, config)
            # Imitação é declarada e acompanha o reforço; não é descoberta de controle do zero.
            imitation = F.mse_loss(output["thrust"], teacher_thrust) + 0.1 * F.binary_cross_entropy_with_logits(output["grip_logit"], teacher_grip)
            distribution = torch.distributions.Normal(output["thrust"], 0.12)
            action = distribution.sample()
            log_probability = distribution.log_prob(action).sum(-1).reshape(config.batch, 2).mean(-1)
            grip = (output["grip_logit"].detach() > 0).reshape(config.batch, 2)
            alive = (~arena.done).float()
            choices = torch.stack((arena.target, selection), -1)
            _, reward, done, _ = arena.step(action.detach().reshape(config.batch, 2, 2), grip, messages, choices)
            with torch.no_grad():
                next_observation, _, _ = sensory_input(arena, language)
                next_value = brain(next_observation, state, config.dt)["value"].reshape(config.batch, 2).mean(-1)
                target = reward + 0.95 * next_value * (~done)
            value = output["value"].reshape(config.batch, 2).mean(-1)
            advantage = target - value
            actor = -(log_probability * advantage.detach() * alive).sum() / alive.sum().clamp_min(1)
            critic = (advantage.square() * alive).sum() / alive.sum().clamp_min(1)
            loss = imitation + 0.03 * actor + 0.1 * critic
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach()))
            rewards.append(float(reward.mean()))
            if step % 10 == 0:
                journal.record("world_step", episode=episode, step=step, world=arena.snapshot(), topology=brain.topology())
            if bool(done.all()): break
        success_rate = float(arena.success.float().mean())
        event = None
        event = brain.adapt(1 - success_rate, float(arena.energy.mean()))
        if event:
            # Resize substitui Parameters; recriar Adam é obrigatório.
            optimizer = torch.optim.Adam(brain.parameters(), lr=0.002)
        if (episode + 1) % 4 == 0:
            brain.rewire(2)
        record = {"episode": episode + 1, "success_rate": success_rate, "nodes": brain.nodes,
                  "loss": sum(losses) / len(losses), "reward": sum(rewards), "morphology": event}
        history.append(record)
        journal.record("world_episode", **record, topology=brain.topology(), world=arena.snapshot())
        print(f"world episode={episode+1} success={success_rate:.3f} nodes={brain.nodes}", flush=True)
    evaluations = {}
    for channel in ("intact", "muted", "shuffled"):
        evaluations[channel] = evaluate_world(brain, language, config, episodes=128, seed=seed + 10_000, channel=channel)
    evaluations["solo"] = evaluate_world(brain, language, config, episodes=128, seed=seed + 10_000, solo=True)
    report = {"training": history, "evaluation": evaluations, "morphology_events": brain.events,
              "config": asdict(config), "curriculum": "Navegação por imitação + actor-critic; sinais treinados no jogo referencial"}
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "topology.json", brain.topology())
    torch.save({"schema": 2, "brain": brain.export(), "optimizer": optimizer.state_dict(),
                "arena_config": asdict(config), "episode": episodes, "torch_rng": torch.get_rng_state()}, directory / "world.pt")
    return brain, report
