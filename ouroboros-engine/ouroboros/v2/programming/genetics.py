"""Programação genética de árvores abertas e mutações de bytecode.

Terminais e operadores formam uma gramática, não um catálogo de soluções.
Novas composições são construídas por cruzamento e mutação de subárvores.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from ..vm import Instruction, Program
from .evaluation import evaluate


OPERATORS = ("ADD", "SUB", "MUL", "MIN", "MAX")


@dataclass(frozen=True)
class Tree:
    op: str
    value: int = 0
    left: Tree | None = None
    right: Tree | None = None

    def nodes(self):
        yield ()
        if self.left:
            for path in self.left.nodes(): yield (0,) + path
        if self.right:
            for path in self.right.nodes(): yield (1,) + path

    def subtree(self, path):
        if not path: return self
        child = self.left if path[0] == 0 else self.right
        return child.subtree(path[1:])

    def replace(self, path, replacement):
        if not path: return replacement
        if path[0] == 0:
            return Tree(self.op, self.value, self.left.replace(path[1:], replacement), self.right)
        return Tree(self.op, self.value, self.left, self.right.replace(path[1:], replacement))

    @property
    def depth(self):
        return 1 + max(self.left.depth if self.left else 0, self.right.depth if self.right else 0)

    def instructions(self):
        if self.op in {"INPUT", "PUSH"}:
            return [Instruction(self.op, self.value)]
        return self.left.instructions() + self.right.instructions() + [Instruction(self.op)]

    def program(self):
        return Program(tuple(self.instructions() + [Instruction("OUTPUT"), Instruction("HALT")]), "genetic_generated")

    def expression(self):
        if self.op == "INPUT": return ("x", "y")[self.value]
        if self.op == "PUSH": return str(self.value)
        if self.op in {"MIN", "MAX"}: return f"{self.op.lower()}({self.left.expression()}, {self.right.expression()})"
        operator = {"ADD": "+", "SUB": "-", "MUL": "*"}[self.op]
        return f"({self.left.expression()} {operator} {self.right.expression()})"


def random_tree(rng, depth):
    if depth <= 1 or rng.random() < 0.28:
        return Tree("INPUT", rng.randrange(2)) if rng.random() < 0.65 else Tree("PUSH", rng.randint(-3, 3))
    return Tree(rng.choice(OPERATORS), left=random_tree(rng, depth - 1), right=random_tree(rng, depth - 1))


class GeneticProgrammer:
    def __init__(self, population=96, generations=100, max_depth=7, seed=42):
        if population < 8 or generations < 1 or max_depth < 2:
            raise ValueError("População >= 8, gerações >= 1 e profundidade >= 2")
        self.population, self.generations, self.max_depth = population, generations, max_depth
        self.rng = random.Random(seed)
        self.cache = {}

    def search(self, task, callback=None):
        self.cache.clear()
        population = [random_tree(self.rng, self.rng.randint(2, min(5, self.max_depth))) for _ in range(self.population)]
        history, champion = [], None

        def score(tree):
            key = tree
            if key not in self.cache:
                evaluation = evaluate(tree.program(), task.train)
                # Complexidade só desempata; nenhum ganho de velocidade compensa incorreção.
                self.cache[key] = (evaluation.error, sum(1 for _ in tree.nodes()), evaluation)
            error, size, evaluation = self.cache[key]
            return error + size * 0.00001, evaluation

        for generation in range(self.generations):
            ranked = sorted(population, key=lambda tree: score(tree)[0])
            champion = ranked[0]
            evaluation = score(champion)[1]
            record = dict(generation=generation, error=evaluation.error, solved_training=evaluation.solved,
                          nodes=sum(1 for _ in champion.nodes()), expression=champion.expression())
            history.append(record)
            if callback: callback(record)
            if evaluation.solved: break
            children = ranked[:max(2, self.population // 12)]

            def tournament():
                return min(self.rng.sample(ranked, 4), key=lambda tree: score(tree)[0])

            while len(children) < self.population:
                parent = tournament()
                path = self.rng.choice(list(parent.nodes()))
                if self.rng.random() < 0.55:
                    donor = tournament()
                    replacement = donor.subtree(self.rng.choice(list(donor.nodes())))
                else:
                    replacement = random_tree(self.rng, self.rng.randint(1, 4))
                child = parent.replace(path, replacement)
                if child.depth <= self.max_depth:
                    children.append(child)
                elif self.rng.random() < 0.1:
                    children.append(random_tree(self.rng, 3))
            population = children
        return champion.program(), history


def mutate_bytecode(program, rng, probability=0.15):
    """Mutação geral para candidatos neurais, incluindo operandos e fluxo.

    Saltos absolutos mantêm limites estáticos. Falhas de pilha ou loops são
    avaliados pela VM e não derrubam o processo de busca.
    """
    instructions = list(program.code)
    for i, instruction in enumerate(instructions):
        if rng.random() >= probability: continue
        op, arg = instruction.op, instruction.arg
        if op in {"JMP", "JZ", "JNZ"}:
            instructions[i] = Instruction(op, rng.randrange(len(instructions)))
        elif op == "PUSH":
            instructions[i] = Instruction(op, int(arg) + rng.choice((-2, -1, 1, 2)))
        elif op in OPERATORS:
            instructions[i] = Instruction(rng.choice(OPERATORS))
        elif op == "INPUT":
            instructions[i] = Instruction(op, rng.randrange(2))
    return Program(tuple(instructions), "bytecode_mutation").validate()
