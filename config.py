from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class CCVRLConfig:
    """Central configuration for CCVRL."""

    task_name: str = "corevalue"
    num_labels: int = 8
    num_roles: int = 7
    hidden_size: int = 768
    max_length: int = 256

    prototype_count: int = 6
    prototype_temperature: float = 0.07
    action_similarity_threshold: float = 0.84
    target_difference_threshold: float = 1.0
    reversal_margin: float = 0.35
    stability_core_threshold: float = 0.80
    stability_background_threshold: float = 0.50

    ambiguity_entropy_weight: float = 0.40
    routing_threshold: float = 0.50
    gate_temperature: float = 0.10
    candidate_count: int = 4
    relation_rounds: int = 2
    compute_budget_ratio: float = 0.35

    direction_loss_weight: float = 1.0
    reversal_loss_weight: float = 0.10
    stability_loss_weight: float = 0.05
    budget_loss_weight: float = 0.10

    light_path_cost: float = 1.0
    deep_path_incremental_cost: float = 1.0

    dropout: float = 0.10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 10
    gradient_clip_norm: float = 1.0
    seed: int = 42

    @classmethod
    def for_task(cls, task_name: str, hidden_size: int = 768) -> "CCVRLConfig":
        task = task_name.strip().lower()
        if task == "corevalue":
            return cls(
                task_name="corevalue",
                hidden_size=hidden_size,
                prototype_count=6,
                action_similarity_threshold=0.84,
            )
        if task in {"c-varc-8", "c_varc_8", "cvarc8"}:
            return cls(
                task_name="c-varc-8",
                hidden_size=hidden_size,
                prototype_count=8,
                action_similarity_threshold=0.90,
            )
        raise ValueError(f"Unsupported task: {task_name}")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
