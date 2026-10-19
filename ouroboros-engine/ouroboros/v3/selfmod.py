"""Reversible equation revisions with an external evaluator and immutable contract.

This deliberately changes neuron definitions, not Python source or evaluators.
"""
import json
from pathlib import Path

from . import fingerprint
from .experiments import atomic_json
from .neuron_dsl import parse


class RevisionWorkspace:
    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = json.loads(json.dumps(contract))
        self.contract_hash = fingerprint(self.contract)
        self.path = self.directory / "revisions.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["contract"] != self.contract_hash:
                raise ValueError("Evaluator contract changed")
        else:
            self.state = {"contract": self.contract_hash, "active": None, "history": []}
            atomic_json(self.path, self.state)

    def propose(self, expression, parent=None):
        expression = parse(expression.to_dict())
        proposal = {"expression": expression.to_dict(), "parent": parent, "contract": self.contract_hash}
        proposal["identity"] = fingerprint(proposal)
        atomic_json(self.directory / f"{proposal['identity']}.json", proposal)
        return proposal

    def evaluate_and_promote(self, proposal, evaluator):
        payload = {key: proposal[key] for key in ("expression", "parent", "contract")}
        if fingerprint(payload) != proposal.get("identity") or proposal["contract"] != self.contract_hash:
            raise ValueError("Tampered proposal")
        parse(proposal["expression"])
        if proposal["parent"] != self.state["active"]:
            raise ValueError("Stale revision parent")
        decision = evaluator(parse(proposal["expression"]), json.loads(json.dumps(self.contract)))
        if type(decision) is not dict or type(decision.get("accepted")) is not bool:
            raise ValueError("Evaluator must return an explicit decision")
        # Serialize before changing active state so malformed evidence cannot partially commit.
        fingerprint(decision)
        if not decision["accepted"]:
            return decision
        event = {"previous": self.state["active"], "next": proposal["identity"], "evidence": decision}
        self.state["history"].append(event)
        self.state["active"] = proposal["identity"]
        atomic_json(self.path, self.state)
        return decision

    def rollback(self):
        for event in reversed(self.state["history"]):
            if not event.get("reverted"):
                self.state["active"] = event["previous"]
                event["reverted"] = True
                atomic_json(self.path, self.state)
                return True
        return False
