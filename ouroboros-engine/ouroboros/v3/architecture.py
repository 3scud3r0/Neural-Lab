"""Heterogeneous recurrent regions and state-preserving structural mutations."""
from dataclasses import dataclass
import copy
import random

import torch
from torch import nn

from .neurons import CellState, InventedCell


@dataclass
class Region:
    name: str
    hidden: int
    expression: object


class HeterogeneousBrain(nn.Module):
    def __init__(self, inputs, outputs, regions):
        super().__init__()
        if not regions or len({r.name for r in regions}) != len(regions):
            raise ValueError("Unique nonempty brain regions required")
        self.specs = list(regions)
        self.regions = nn.ModuleDict({r.name: InventedCell(inputs, r.hidden, r.expression) for r in regions})
        self.readout = nn.Linear(sum(r.hidden for r in regions), outputs)

    def forward(self, value, states=None):
        states, following, outputs = states or {}, {}, []
        for name, region in self.regions.items():
            hidden, following[name] = region(value, states.get(name))
            outputs.append(hidden)
        return self.readout(torch.cat(outputs, -1)), following


def _overlap_copy(destination, source):
    if destination.ndim != source.ndim:
        raise ValueError("Cannot migrate tensor rank")
    slices = tuple(slice(0, min(a, b)) for a, b in zip(destination.shape, source.shape))
    destination[slices].copy_(source[slices])


def resize_region(brain, name, hidden, optimizer=None, states=None):
    if name not in brain.regions or not 1 <= hidden <= 128:
        raise ValueError("Invalid region mutation")
    if optimizer is not None and (not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1):
        raise ValueError("Only single-group Adam migration is supported")
    old_named = dict(brain.named_parameters())
    old_region, old_readout = brain.regions[name], brain.readout
    device = old_region.decay.device
    new_region = InventedCell(old_region.inputs, hidden, old_region.expression, old_region.spiking).to(device)
    offset = sum(region.hidden for region in brain.specs[:next(i for i, r in enumerate(brain.specs) if r.name == name)])
    previous, total = old_region.hidden, old_readout.in_features - old_region.hidden + hidden
    new_readout = nn.Linear(total, old_readout.out_features).to(device)
    with torch.no_grad():
        for key, value in new_region.state_dict().items():
            _overlap_copy(value, old_region.state_dict()[key])
        new_readout.bias.copy_(old_readout.bias)
        new_readout.weight[:, :offset].copy_(old_readout.weight[:, :offset])
        kept = min(previous, hidden)
        new_readout.weight[:, offset:offset + hidden].zero_()
        new_readout.weight[:, offset:offset + kept].copy_(old_readout.weight[:, offset:offset + kept])
        new_readout.weight[:, offset + hidden:].copy_(old_readout.weight[:, offset + previous:])
    brain.regions[name], brain.readout = new_region, new_readout
    brain.specs = [Region(r.name, hidden if r.name == name else r.hidden, r.expression) for r in brain.specs]
    migrated_optimizer = None
    if optimizer is not None:
        migrated_optimizer = torch.optim.Adam(brain.parameters(), **optimizer.defaults)
        for key, parameter in brain.named_parameters():
            old_parameter = old_named[key]
            old_state = optimizer.state.get(old_parameter, {})
            new_state = {}
            for state_name, tensor in old_state.items():
                if not torch.is_tensor(tensor) or tensor.ndim == 0:
                    new_state[state_name] = copy.deepcopy(tensor)
                elif key == "readout.weight":
                    value = torch.zeros_like(parameter)
                    value[:, :offset] = tensor[:, :offset]
                    value[:, offset:offset + kept] = tensor[:, offset:offset + kept]
                    value[:, offset + hidden:] = tensor[:, offset + previous:]
                    new_state[state_name] = value
                else:
                    value = torch.zeros_like(parameter)
                    _overlap_copy(value, tensor)
                    new_state[state_name] = value
            if new_state:
                migrated_optimizer.state[parameter] = new_state
    migrated_states = None
    if states is not None:
        # A structural edit is a truncated-backpropagation boundary. Old graph
        # references must not survive parameter replacement or an optimizer step.
        migrated_states = {key: value.detach() for key, value in states.items()}
        if name in states:
            old = states[name].detach()
            values = []
            for tensor in (old.h, old.memory, old.adaptation, old.phase):
                value = tensor.new_zeros(tensor.shape[0], hidden)
                value[:, :kept] = tensor[:, :kept]
                values.append(value)
            migrated_states[name] = CellState(*values)
    return migrated_optimizer, migrated_states


def development_decision(stress, energy, hidden, max_hidden=128):
    if energy < 0.2 and hidden > 4:
        return hidden - 1
    if stress > 0.7 and energy > 0.5 and hidden < max_hidden:
        return hidden + 1
    return hidden


class ArchitectureSequence(nn.Module):
    def __init__(self, regions):
        super().__init__()
        self.brain = HeterogeneousBrain(3, 1, regions)

    def forward(self, inputs):
        state, outputs = None, []
        for value in inputs.unbind(1):
            output, state = self.brain(value, state)
            outputs.append(output)
        return torch.stack(outputs, 1)


def search_architectures(expressions, candidates=8, steps=8, seed=42):
    """Jointly sample region counts, widths and neuron equations on validation."""
    from .benchmarks import evaluate, train_steps
    from .experiments import isolated_seed
    if not expressions or not 1 <= candidates <= 256 or steps < 1:
        raise ValueError("Invalid architecture search budget")
    rng, records = random.Random(seed), []
    for index in range(candidates):
        regions = [Region(f"region_{r}", rng.choice((4, 6, 8, 12)), rng.choice(expressions))
                   for r in range(rng.randint(1, 3))]
        with isolated_seed(seed):
            model = ArchitectureSequence(regions)
            train_steps(model, steps, seed, batch=8, length=8)
            result = evaluate(model, "validation", seed, batch=8, length=8)
            records.append({"index": index, "regions": [{"name": r.name, "hidden": r.hidden,
                              "expression": r.expression.to_dict()} for r in regions],
                            "validation": result, "parameters": sum(p.numel() for p in model.parameters())})
    selected = min(records, key=lambda record: record["validation"]["mean"])
    return {"selected": selected["index"], "candidates": records,
            "scope": "budgeted random joint search; no architecture superiority claim"}
