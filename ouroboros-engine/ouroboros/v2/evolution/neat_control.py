"""NEAT upstream evolui um controlador; exportação JSON evita pickle livre."""
from __future__ import annotations

import json
from pathlib import Path
import random

import neat
from neat.reporting import BaseReporter
import numpy as np
import torch

from ..common import ROOT, Journal, atomic_json
from ..world.training import navigation_teacher
from ..world.arena import ArenaConfig


def control_dataset(seed, count=64):
    generator = torch.Generator().manual_seed(seed)
    observations = torch.rand(count, 12, generator=generator) * 2 - 1
    observations[:, 4] = (observations[:, 4] + 1) / 2
    observations[:, 5] = (torch.arange(count) % 2).float()
    observations[:, 8] = (observations[:, 8] + 1) / 2
    observations[:, 11] = 1
    targets, _ = navigation_teacher(observations, ArenaConfig())
    return observations.numpy().tolist(), targets.numpy()


class StructuralReporter(BaseReporter):
    def __init__(self, journal):
        self.generation, self.history, self.journal = 0, [], journal

    def start_generation(self, generation): self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        values = list(population.values())
        record = {"generation": self.generation, "fitness": best_genome.fitness,
                  "winner_nodes": len(best_genome.nodes),
                  "winner_edges": sum(c.enabled for c in best_genome.connections.values()),
                  "population_max_nodes": max(len(g.nodes) for g in values),
                  "population_min_nodes": min(len(g.nodes) for g in values)}
        self.history.append(record)
        self.journal.record("neat", **record)


def export_network(network):
    return {"inputs": network.input_nodes, "outputs": network.output_nodes,
            "node_evals": [{"node": n, "activation": activation.__name__, "aggregation": aggregation.__name__,
                            "bias": bias, "response": response, "links": links}
                           for n, activation, aggregation, bias, response, links in network.node_evals]}


def load_network(data):
    from neat.activations import tanh_activation
    from neat.aggregations import sum_aggregation
    activations, aggregations = {"tanh_activation": tanh_activation}, {"sum_aggregation": sum_aggregation}
    evaluations = [(item["node"], activations[item["activation"]], aggregations[item["aggregation"]],
                    item["bias"], item["response"], item["links"]) for item in data["node_evals"]]
    return neat.nn.FeedForwardNetwork(data["inputs"], data["outputs"], evaluations)


def evolve_controller(directory, generations=20, seed=42, config_path=None):
    if generations < 1: raise ValueError("Gerações devem ser positivas")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = config_path or ROOT / "configs/v2/neat-control.ini"
    random.seed(seed)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(config_path))
    inputs, targets = control_dataset(seed)

    def fitness(genomes, current_config):
        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, current_config)
            outputs = np.array([network.activate(row) for row in inputs])
            error = float(np.square(outputs - targets).mean())
            genome.fitness = 1 / (1 + error)

    population = neat.Population(config)
    reporter = StructuralReporter(Journal(directory))
    population.add_reporter(reporter)
    winner = population.run(fitness, generations)
    network = neat.nn.FeedForwardNetwork.create(winner, config)
    validation_inputs, validation_targets = control_dataset(seed + 10_000, 128)
    validation_error = float(np.square(np.array([network.activate(row) for row in validation_inputs]) - validation_targets).mean())
    report = {"history": reporter.history, "validation_mse": validation_error, "nodes": len(winner.nodes),
              "connections": sum(c.enabled for c in winner.connections.values()),
              "structural_variation_observed": any(row["population_max_nodes"] > 2 for row in reporter.history),
              "task": "Imitação de controle de navegação; não mede inteligência geral"}
    atomic_json(directory / "controller.json", export_network(network))
    atomic_json(directory / "report.json", report)
    return network, report
