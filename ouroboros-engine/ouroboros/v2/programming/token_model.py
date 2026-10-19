"""Decoder autorregressivo de instruções e operandos, sem catálogo de programas."""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.nn import functional as F

from ..vm.language import ARGUMENT_OPS, STACK_EFFECT, Instruction, Program
from .evaluation import evaluate
from .tasks import SOURCES, make_task


class BytecodeTokenizer:
    def __init__(self, smallest=-16, largest=255):
        self.smallest, self.largest = smallest, largest
        self.tokens = ["PAD", "BOS", "EOS"] + sorted(STACK_EFFECT) + [f"#{i}" for i in range(smallest, largest + 1)]
        self.lookup = {token: i for i, token in enumerate(self.tokens)}
        self.pad, self.bos, self.eos = 0, 1, 2
        self.opcode_ids = [self.lookup[op] for op in STACK_EFFECT]
        self.argument_ids = [self.lookup[f"#{i}"] for i in range(smallest, largest + 1)]

    def encode(self, program):
        sequence = [self.bos]
        for instruction in program.code:
            sequence.append(self.lookup[instruction.op])
            if instruction.arg is not None:
                if instruction.arg != int(instruction.arg):
                    raise ValueError("O decoder usa operandos inteiros; a VM aceita floats via assembly")
                token = f"#{int(instruction.arg)}"
                if token not in self.lookup: raise ValueError("Operando fora do vocabulário neural")
                sequence.append(self.lookup[token])
        return sequence + [self.eos]

    def decode(self, sequence):
        instructions, pending = [], None
        for index in sequence:
            token = self.tokens[int(index)]
            if token in {"BOS", "PAD"}: continue
            if token == "EOS": break
            if pending is not None:
                if not token.startswith("#"): raise ValueError("Esperado operando")
                instructions.append(Instruction(pending, int(token[1:])))
                pending = None
            elif token in STACK_EFFECT:
                if token in ARGUMENT_OPS: pending = token
                else: instructions.append(Instruction(token))
            else:
                raise ValueError("Esperado opcode")
        if pending is not None: raise ValueError("Programa termina com operando ausente")
        return Program(tuple(instructions), "neural_generated").validate()

    def allowed(self, pending, device):
        mask = torch.zeros(len(self.tokens), dtype=torch.bool, device=device)
        mask[self.argument_ids if pending else self.opcode_ids + [self.eos]] = True
        if pending and pending != "PUSH":
            mask[self.argument_ids[:max(0, -self.smallest)]] = False
        if pending == "INPUT":
            mask.zero_()
            mask[[self.lookup[f"#{i}"] for i in range(8)]] = True
        return mask


class ExampleEncoder(nn.Module):
    """Deep Sets: exemplos não possuem ordem semântica para definir uma função."""

    def __init__(self, hidden):
        super().__init__()
        self.case = nn.Sequential(nn.Linear(9, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.pool = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())

    def forward(self, contexts):
        rows = contexts.reshape(contexts.shape[0], -1, 9)
        features = self.case(rows)
        return self.pool(torch.cat((features.mean(1), features.max(1).values), -1))


class TokenProgrammer(nn.Module):
    def __init__(self, hidden=128, context_size=72):
        super().__init__()
        self.tokenizer = BytecodeTokenizer()
        self.hidden_size, self.context_size = hidden, context_size
        if context_size < 9 or context_size % 9: raise ValueError("Contexto precisa conter exemplos de nove features")
        self.encoder = ExampleEncoder(hidden)
        self.embedding = nn.Embedding(len(self.tokenizer.tokens), hidden)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(2 * hidden, len(self.tokenizer.tokens))

    def forward(self, contexts, prefix):
        context = self.encoder(contexts)
        conditioned = self.embedding(prefix) + context[:, None]
        states, _ = self.decoder(conditioned, context.unsqueeze(0))
        return self.head(torch.cat((states, context[:, None].expand(-1, states.shape[1], -1)), -1))

    def generate(self, context, max_tokens=160, temperature=0.8, greedy=False):
        if max_tokens < 2 or temperature <= 0: raise ValueError("Parâmetros de geração inválidos")
        encoded = self.encoder(context.unsqueeze(0))
        hidden = encoded.unsqueeze(0)
        previous = torch.tensor([[self.tokenizer.bos]], device=context.device)
        sequence, log_probabilities, entropies, pending = [self.tokenizer.bos], [], [], None
        for _ in range(max_tokens):
            state, hidden = self.decoder(self.embedding(previous) + encoded[:, None], hidden)
            logits = self.head(torch.cat((state[0, 0], encoded[0]), -1)) / temperature
            mask = self.tokenizer.allowed(pending, context.device)
            if len(sequence) == 1: mask[self.tokenizer.eos] = False
            logits = logits.masked_fill(~mask, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            chosen = logits.argmax() if greedy else distribution.sample()
            log_probabilities.append(distribution.log_prob(chosen))
            entropies.append(distribution.entropy())
            token_id = int(chosen.item())
            sequence.append(token_id)
            token = self.tokenizer.tokens[token_id]
            if token == "EOS": break
            pending = None if pending is not None else token if token in ARGUMENT_OPS else None
            previous = chosen.reshape(1, 1)
        try:
            program = self.tokenizer.decode(sequence)
        except ValueError:
            program = None
        return program, torch.stack(log_probabilities).sum(), torch.stack(entropies).mean(), sequence


def supervised_bootstrap(model, tasks, epochs=200, learning_rate=0.003, seed=42, callback=None):
    if epochs < 1 or not tasks: raise ValueError("Treino exige tarefas e épocas positivas")
    rng = random.Random(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sequences = [model.tokenizer.encode(task.reference_program()) for task in tasks]
    width = max(map(len, sequences))
    tokens = torch.full((len(tasks), width), model.tokenizer.pad, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences): tokens[i, :len(sequence)] = torch.tensor(sequence, device=device)
    contexts = torch.stack([task.context(device) for task in tasks])
    history = []
    best_state, best_score = None, (-1, -1.0, -float("inf"))
    model.train()
    for epoch in range(epochs):
        # Novos exemplos de treino em cada época, no domínio de treino.
        # A ordem não é informação semântica; gerar novos valores evita memorizar 11 bags.
        public_contexts = []
        for task in tasks:
            if task.name in SOURCES:
                augmented = make_task(task.name, seed=rng.randrange(2**31), counts=(8, 0, 0))
                public_contexts.append(augmented.context(device))
            else:
                public_contexts.append(task.context(device))
        contexts = torch.stack(public_contexts)
        logits = model(contexts, tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=model.tokenizer.pad)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            record = {"epoch": epoch + 1, "loss": float(loss.detach())}
            # Seleção usa SOMENTE validação. Casos finais são abertos depois.
            with torch.no_grad():
                validation_results = []
                for task in tasks:
                    program, *_ = model.generate(task.context(device), greedy=True)
                    validation_results.append(evaluate(program, task.validation) if program else None)
                solved = sum(bool(result and result.solved) for result in validation_results)
                fractional = sum(result.passed / result.total if result else 0 for result in validation_results)
            record["validation_families_solved"] = solved
            score = (solved, fractional, -record["loss"])
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            history.append(record)
            if callback: callback(record)
            if solved == len(tasks):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def reinforce(model, task, attempts=32, learning_rate=0.0002):
    """Feedback usa treino; os testes finais nunca participam da atualização."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    baseline, history = 0.0, []
    for attempt in range(attempts):
        program, log_prob, entropy, _ = model.generate(task.context(device))
        evaluation = evaluate(program, task.train) if program else None
        reward = evaluation.reward if evaluation else -1.25
        loss = -(reward - baseline) * log_prob - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        baseline = 0.9 * baseline + 0.1 * reward
        history.append({"attempt": attempt + 1, "reward": reward,
                        "valid": program is not None, "solved_training": bool(evaluation and evaluation.solved)})
    return history
