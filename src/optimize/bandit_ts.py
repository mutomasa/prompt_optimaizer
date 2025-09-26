"""Thompson Sampling utilities for prompt knob optimisation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ArmState:
    """Statistics tracked for each knob configuration."""

    knobs: Dict[str, Any]
    wins: float = 1.0
    plays: int = 0
    alpha: float = 1.0
    beta: float = 1.0
    history: List[Dict[str, float]] = field(default_factory=list)

    def sample(self) -> float:
        """Draw a Thompson sample from the Beta posterior."""

        return random.betavariate(self.alpha, self.beta)

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.plays)

    def record(self, reward: float, metadata: Dict[str, float] | None = None) -> None:
        self.plays += 1
        self.wins += reward
        self.alpha += reward
        self.beta += max(0.0, 1.0 - reward)
        if metadata is not None:
            self.history.append(metadata)


def select_top_arms(arms: List[ArmState], k: int = 2) -> List[ArmState]:
    """Select `k` arms with highest Thompson samples."""

    scored = [(arm.sample(), arm) for arm in arms]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [arm for _, arm in scored[:k]]


def mutate_knobs(space: Dict[str, List[Any]], base: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly mutate a knob configuration."""

    if not space:
        return base
    knob = random.choice(list(space.keys()))
    options = space[knob]
    if not options:
        return base
    new_value = random.choice(options)
    mutated = {**base, knob: new_value}
    return mutated
