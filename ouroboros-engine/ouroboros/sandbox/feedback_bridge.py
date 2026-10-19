from dataclasses import dataclass
import math


@dataclass
class Feedback:
    reward: float
    shock: float
    solved: bool


def feedback_from_result(result, syntax_penalty=1.0):
    if not result.ok:
        return Feedback(-syntax_penalty, 1.0, False)
    solved = result.error < 1e-6
    # Correção determina recompensa; uma saída vazia nunca é sucesso.
    quality = math.exp(-min(result.error, 100))
    reward = 2 * quality - 1 if result.values else -1.0
    return Feedback(reward, min(1.0, result.error), solved and bool(result.values))
