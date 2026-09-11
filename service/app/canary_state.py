"""In-memory canary state: two model slots, current and next.

This is the only place that knows about the canary split. Routes call
into this module, they never touch mlflow or a raw model object
directly, which keeps the routing logic testable with a stub model.
"""
import random
from dataclasses import dataclass, field
from typing import Literal

SlotName = Literal["current", "next"]


@dataclass
class ModelSlot:
    model: object | None = None
    version: str | None = None
    request_count: int = 0

    @property
    def loaded(self) -> bool:
        return self.model is not None


@dataclass
class CanaryState:
    probability_current: float
    current: ModelSlot = field(default_factory=ModelSlot)
    next: ModelSlot = field(default_factory=ModelSlot)

    def set_both(self, model: object, version: str) -> None:
        """Used at startup: current and next start out identical."""
        self.current = ModelSlot(model=model, version=version)
        self.next = ModelSlot(model=model, version=version)

    def set_next(self, model: object, version: str) -> None:
        """Used by /update-model: only the next slot changes."""
        self.next = ModelSlot(model=model, version=version)

    def promote_next(self) -> None:
        """Used by /accept-next-model: next becomes current, both equal."""
        self.current = ModelSlot(model=self.next.model, version=self.next.version)

    def choose_slot(self, roll: float | None = None) -> SlotName:
        """Pick current with probability p, next with probability 1-p.

        roll is injectable for deterministic tests (p=0 or p=1 alone is
        not always enough once random.random() is involved elsewhere).
        """
        value = random.random() if roll is None else roll
        return "current" if value < self.probability_current else "next"

    def slot(self, name: SlotName) -> ModelSlot:
        return self.current if name == "current" else self.next

    def record_request(self, name: SlotName) -> None:
        self.slot(name).request_count += 1
