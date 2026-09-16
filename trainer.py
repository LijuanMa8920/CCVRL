from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from config import CCVRLConfig
from metrics import evaluate_predictions
from objectives import compute_ccvrl_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class CCVRLTrainer:
    """Training and evaluation loop for CCVRL."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: CCVRLConfig,
        device: Optional[str] = None,
        use_amp: bool = True,
        reversal_pairs: Optional[torch.Tensor] = None,
        stability_pairs: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.reversal_pairs = reversal_pairs.long().cpu() if reversal_pairs is not None else None
        self.stability_pairs = stability_pairs.long().cpu() if stability_pairs is not None else None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        set_seed(config.seed)


    @staticmethod
    def _localize_pairs(global_pairs: Optional[torch.Tensor], sample_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if global_pairs is None:
            return None
        mapping = {int(global_id): local_id for local_id, global_id in enumerate(sample_ids.cpu().tolist())}
        localized = []
        for first, second in global_pairs.tolist():
            if first in mapping and second in mapping:
                localized.append([mapping[first], mapping[second]])
        if not localized:
            return torch.empty((0, 2), dtype=torch.long, device=sample_ids.device)
        return torch.tensor(localized, dtype=torch.long, device=sample_ids.device)

    def train_epoch(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        steps = 0

        for batch in loader:
            batch = move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    hard_routing=False,
                )
                reversal_pairs = self._localize_pairs(self.reversal_pairs, batch["sample_ids"]) if "sample_ids" in batch else None
                stability_pairs = self._localize_pairs(self.stability_pairs, batch["sample_ids"]) if "sample_ids" in batch else None
                loss_dict = compute_ccvrl_loss(
                    outputs,
                    labels=batch["labels"],
                    directions=batch.get("directions"),
                    config=self.config,
                    reversal_pairs=reversal_pairs,
                    stability_pairs=stability_pairs,
                )
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            steps += 1
            for key, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())

        if steps == 0:
            raise ValueError("Training loader produced zero batches")
        return {key: value / steps for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        label_frequency: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        self.model.eval()
        probabilities = []
        labels = []
        ambiguity = []
        direction_logits = []
        directions = []
        hard_gates = []

        for batch in loader:
            batch = move_batch(batch, self.device)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                hard_routing=True,
            )
            probabilities.append(outputs["category_probabilities"].cpu())
            labels.append(batch["labels"].cpu())
            ambiguity.append(outputs["ambiguity"].cpu())
            direction_logits.append(outputs["direction_logits"].cpu())
            directions.append(batch["directions"].cpu())
            hard_gates.append(outputs["hard_gate"].cpu())

        if not probabilities:
            raise ValueError("Evaluation loader produced zero batches")

        probability_tensor = torch.cat(probabilities, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        ambiguity_tensor = torch.cat(ambiguity, dim=0)
        direction_tensor = torch.cat(direction_logits, dim=0)
        direction_target = torch.cat(directions, dim=0)
        gate_tensor = torch.cat(hard_gates, dim=0)

        metrics = evaluate_predictions(
            probability_tensor,
            label_tensor,
            ambiguity_tensor,
            threshold=threshold,
            direction_logits=direction_tensor,
            directions=direction_target,
            label_frequency=label_frequency,
            task_name=self.config.task_name,
        )
        metrics["trigger_rate"] = float(gate_tensor.mean().item())
        return metrics

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        validation_loader: Iterable[Dict[str, torch.Tensor]],
        epochs: Optional[int] = None,
        label_frequency: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        total_epochs = epochs or self.config.epochs
        monitor = "joint_f1" if self.config.task_name == "corevalue" else "macro_f1"
        best_score = float("-inf")
        best_state = None
        history = []

        for epoch in range(1, total_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            validation_metrics = self.evaluate(
                validation_loader,
                label_frequency=label_frequency,
            )
            score = validation_metrics.get(monitor, float("-inf"))
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is None:
            raise RuntimeError("No valid checkpoint was selected")
        self.model.load_state_dict(best_state)
        return {"best_score": best_score, "monitor": monitor, "history": history}

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    @staticmethod
    def save_metrics(metrics: Dict[str, object], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
