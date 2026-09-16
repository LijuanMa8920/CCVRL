from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler


ROLE_NAMES: Tuple[str, ...] = (
    "subject",
    "action",
    "precondition",
    "object",
    "manner",
    "time",
    "location",
)


class MultiLabelJsonlDataset(Dataset):
    """JSONL dataset for eight-label category prediction and optional directions."""

    def __init__(self, path: str | Path, num_labels: int = 8) -> None:
        self.path = Path(path)
        self.num_labels = num_labels
        self.records = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "text" not in item or "labels" not in item:
                    raise ValueError(f"Missing text or labels at {path}:{line_number}")
                records.append(item)
        if not records:
            raise ValueError(f"Dataset is empty: {path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = dict(self.records[index])
        item["_sample_id"] = index
        return item

    def label_frequency(self) -> torch.Tensor:
        counts = torch.zeros(self.num_labels, dtype=torch.long)
        for item in self.records:
            labels = _to_multihot(item["labels"], self.num_labels)
            counts += labels.to(torch.long)
        return counts


def _to_multihot(value: object, num_labels: int) -> torch.Tensor:
    if isinstance(value, list):
        if len(value) == num_labels and all(isinstance(x, (int, float, bool)) for x in value):
            tensor = torch.tensor(value, dtype=torch.float32)
            if torch.all((tensor == 0) | (tensor == 1)):
                return tensor
        output = torch.zeros(num_labels, dtype=torch.float32)
        for label_index in value:
            output[int(label_index)] = 1.0
        return output
    raise TypeError("labels must be a multi-hot list or a list of active indices")


def _normalize_direction_value(value: int) -> int:
    if value in (0, 1):
        return value
    if value == -1:
        return 0
    if value == 1:
        return 1
    raise ValueError("Direction values must use {-1, +1} or {0, 1}")


def _to_direction_tensor(value: object, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    output = torch.full((num_labels,), -100, dtype=torch.long)
    if value is None:
        return output

    if isinstance(value, list) and len(value) == num_labels:
        active = labels.bool()
        for idx in active.nonzero(as_tuple=False).flatten().tolist():
            output[idx] = _normalize_direction_value(int(value[idx]))
        return output

    if isinstance(value, dict):
        for key, direction in value.items():
            idx = int(key)
            if labels[idx] > 0:
                output[idx] = _normalize_direction_value(int(direction))
        return output

    raise TypeError("directions must be a length-L list, a label-index dictionary, or null")


class TextBatchCollator:
    """Tokenize text and assemble model-ready tensors."""

    def __init__(self, tokenizer_name: str, max_length: int = 256, num_labels: int = 8) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for TextBatchCollator") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.num_labels = num_labels

    def __call__(self, records: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [str(item["text"]) for item in records]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.stack([_to_multihot(item["labels"], self.num_labels) for item in records])
        directions = torch.stack(
            [
                _to_direction_tensor(item.get("directions"), labels[i], self.num_labels)
                for i, item in enumerate(records)
            ]
        )

        batch: Dict[str, torch.Tensor] = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
            "directions": directions,
            "sample_ids": torch.tensor([int(item.get("_sample_id", i)) for i, item in enumerate(records)], dtype=torch.long),
        }
        if "token_type_ids" in tokens:
            batch["token_type_ids"] = tokens["token_type_ids"]
        return batch



class PairAwareBatchSampler(Sampler[List[int]]):
    """Form batches that preserve as many cached pair endpoints as possible while covering each sample once."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        reversal_pairs: Optional[torch.Tensor] = None,
        stability_pairs: Optional[torch.Tensor] = None,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Pair-aware batching requires batch_size >= 2")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        pair_blocks = []
        for pair_tensor in (reversal_pairs, stability_pairs):
            if pair_tensor is not None and pair_tensor.numel() > 0:
                pair_blocks.extend([tuple(map(int, pair)) for pair in pair_tensor.tolist()])
        self.pairs = pair_blocks

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pairs = list(self.pairs)
        rng.shuffle(pairs)
        remaining = list(range(self.dataset_size))
        rng.shuffle(remaining)
        unused = set(remaining)
        cursor = 0

        while unused:
            batch: List[int] = []
            while pairs and len(batch) + 2 <= self.batch_size:
                first, second = pairs.pop()
                if first in unused and second in unused and first != second:
                    batch.extend([first, second])
                    unused.remove(first)
                    unused.remove(second)

            while len(batch) < self.batch_size and unused:
                while cursor < len(remaining) and remaining[cursor] not in unused:
                    cursor += 1
                if cursor >= len(remaining):
                    break
                index = remaining[cursor]
                cursor += 1
                batch.append(index)
                unused.remove(index)

            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch


def load_pair_tensor(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    tensor = payload["pairs"] if isinstance(payload, dict) and "pairs" in payload else payload
    tensor = torch.as_tensor(tensor, dtype=torch.long)
    if tensor.ndim != 2 or tensor.size(1) != 2:
        raise ValueError("Pair file must have shape [N, 2]")
    return tensor


def spherical_kmeans(
    vectors: torch.Tensor,
    num_clusters: int,
    iterations: int = 40,
    seed: int = 42,
) -> torch.Tensor:
    """Cluster normalized vectors with cosine assignment and mean updates."""

    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [N, D]")
    if vectors.size(0) == 0:
        raise ValueError("vectors must contain at least one row")

    vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
    effective_k = min(num_clusters, vectors.size(0))
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)
    perm = torch.randperm(vectors.size(0), generator=generator, device=vectors.device)
    centers = vectors[perm[:effective_k]].clone()

    for _ in range(iterations):
        similarity = vectors @ centers.transpose(0, 1)
        assignments = similarity.argmax(dim=1)
        updated: List[torch.Tensor] = []
        for cluster_id in range(effective_k):
            members = vectors[assignments == cluster_id]
            if members.numel() == 0:
                replacement = vectors[torch.randint(vectors.size(0), (1,), generator=generator, device=vectors.device)]
                updated.append(replacement.squeeze(0))
            else:
                updated.append(torch.nn.functional.normalize(members.mean(dim=0), dim=0))
        new_centers = torch.stack(updated, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    if effective_k < num_clusters:
        repeats = num_clusters - effective_k
        extra = centers[torch.arange(repeats, device=centers.device) % effective_k]
        centers = torch.cat([centers, extra], dim=0)
    return torch.nn.functional.normalize(centers, dim=-1)


def build_prototype_bank(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    prototypes_per_label: int,
    iterations: int = 40,
    seed: int = 42,
) -> torch.Tensor:
    """Build a [L, K, D] prototype tensor from rule embeddings and multi-hot labels."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    if labels.shape != (embeddings.size(0), num_labels):
        raise ValueError("labels must have shape [N, L]")

    banks: List[torch.Tensor] = []
    for label_idx in range(num_labels):
        selected = embeddings[labels[:, label_idx].bool()]
        if selected.size(0) == 0:
            raise ValueError(f"No rule embedding is available for label {label_idx}")
        banks.append(
            spherical_kmeans(
                selected,
                num_clusters=prototypes_per_label,
                iterations=iterations,
                seed=seed + label_idx,
            )
        )
    return torch.stack(banks, dim=0)


def load_prototype_tensor(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, dict) and "prototypes" in payload:
        return torch.as_tensor(payload["prototypes"], dtype=torch.float32)
    raise ValueError("Prototype file must contain a tensor or a dictionary with key 'prototypes'")
