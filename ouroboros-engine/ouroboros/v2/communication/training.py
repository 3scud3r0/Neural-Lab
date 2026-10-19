"""Treino por sucesso compartilhado e avaliação causal por ablação do canal."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from ..common import Journal, atomic_json, wilson_interval
from .game import CommunicationGame, GameConfig, make_batch


def message_statistics(concepts, messages):
    pairs = list(zip(concepts.tolist(), map(tuple, messages.tolist())))
    joint, concept_count, message_count = Counter(pairs), Counter(), Counter()
    for concept, message in pairs:
        concept_count[concept] += 1
        message_count[message] += 1
    n = len(pairs)
    mutual_information = sum(count / n * math.log2(count * n / (concept_count[c] * message_count[m]))
                             for (c, m), count in joint.items())
    entropy = -sum(count / n * math.log2(count / n) for count in message_count.values())
    lexicon = {}
    for concept in sorted(concept_count):
        choices = {message: count for (c, message), count in joint.items() if c == concept}
        message = max(choices, key=choices.get)
        lexicon[str(concept)] = {"symbols": list(message), "frequency": choices[message] / concept_count[concept]}
    return {"mutual_information_bits": mutual_information, "message_entropy_bits": entropy,
            "unique_messages": len(message_count), "lexicon": lexicon}


@torch.no_grad()
def evaluate_communication(model, trials=1024, seed=1234, partition="train"):
    if trials < 2: raise ValueError("Use pelo menos duas avaliações")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = make_batch(model.config, trials, device, generator, partition)
    message, _ = model.sender(batch.sender_observation, discrete=True)
    permutation = torch.randperm(trials, device=device, generator=generator)
    random_symbols = torch.randint(model.config.vocabulary, (trials, model.config.message_length), device=device, generator=generator)
    channels = {"intact": message, "shuffled": message[permutation], "muted": torch.zeros_like(message),
                "random": F.one_hot(random_symbols, model.config.vocabulary).float()}
    results = {}
    for name, channel in channels.items():
        predictions = model.receiver(channel, batch.receiver_candidates).argmax(-1)
        passed = int((predictions == batch.target_index).sum().item())
        results[name] = dict(correct=passed, total=trials, accuracy=passed / trials,
                             ci95=wilson_interval(passed, trials))
    results["causal_drop_muted"] = results["intact"]["accuracy"] - results["muted"]["accuracy"]
    results["causal_drop_shuffled"] = results["intact"]["accuracy"] - results["shuffled"]["accuracy"]
    results["chance"] = 1 / model.config.candidates
    results["partition"] = partition
    results["statistics"] = message_statistics(batch.target_concept.cpu(), message.argmax(-1).cpu())
    return results


def save_checkpoint(path, model, optimizer, step, generator):
    torch.save({"schema": 2, "config": asdict(model.config), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state(), "batch_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def load_game(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = CommunicationGame(GameConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def train_communication(model, directory: Path, steps=1200, batch_size=128, seed=42,
                        learning_rate=0.003, resume=None, evaluation_trials=1024):
    if steps < 1 or batch_size < 2: raise ValueError("Passos e batch inválidos")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        if saved["config"] != asdict(model.config): raise ValueError("Configuração do checkpoint difere")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        generator.set_state(saved["batch_rng"].cpu())
        if saved["cuda_rng"] and device.type == "cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
    if start >= steps: raise ValueError("steps precisa superar o passo do checkpoint")
    model.train()
    for step in range(start + 1, steps + 1):
        batch = make_batch(model.config, batch_size, device, generator, "train")
        # Temperatura depende do passo absoluto para permitir retomada exata.
        temperature = max(0.4, 1.5 * math.exp(-step / 700))
        scores, messages, sender_logits = model(batch, temperature)
        loss = F.cross_entropy(scores, batch.target_index)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            accuracy = (scores.argmax(-1) == batch.target_index).float().mean().item()
            journal.record("communication", step=step, loss=loss.item(), accuracy=accuracy, temperature=temperature)
            print(f"communication step={step} loss={loss.item():.4f} accuracy={accuracy:.3f}", flush=True)
    model.eval()
    report = {"train_domain": evaluate_communication(model, evaluation_trials, seed + 1, "train"),
              "heldout_combinations": evaluate_communication(model, evaluation_trials, seed + 2, "heldout"),
              "config": asdict(model.config), "steps": steps,
              "training": "Straight-through Gumbel-Softmax; discreto em avaliação; sem rótulos dos símbolos"}
    atomic_json(directory / "report.json", report)
    save_checkpoint(directory / "communication.pt", model, optimizer, steps, generator)
    return report
