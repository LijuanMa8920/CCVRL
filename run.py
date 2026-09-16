from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from config import CCVRLConfig
from data import (
    MultiLabelJsonlDataset,
    PairAwareBatchSampler,
    TextBatchCollator,
    load_pair_tensor,
    load_prototype_tensor,
)
from model import CCVRL, HuggingFaceBackbone
from trainer import CCVRLTrainer, set_seed


def build_loader(
    dataset: MultiLabelJsonlDataset,
    collator: TextBatchCollator,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def build_train_loader(
    dataset: MultiLabelJsonlDataset,
    collator: TextBatchCollator,
    batch_size: int,
    workers: int,
    reversal_pairs: torch.Tensor | None,
    stability_pairs: torch.Tensor | None,
    seed: int,
) -> DataLoader:
    if reversal_pairs is None and stability_pairs is None:
        return build_loader(dataset, collator, batch_size, True, workers)
    sampler = PairAwareBatchSampler(
        dataset_size=len(dataset),
        batch_size=batch_size,
        reversal_pairs=reversal_pairs,
        stability_pairs=stability_pairs,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


def apply_overrides(config: CCVRLConfig, args: argparse.Namespace) -> None:
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.seed = args.seed
    config.stability_core_threshold = args.tau_c
    config.stability_background_threshold = args.tau_b
    config.prototype_temperature = args.prototype_temperature
    config.gate_temperature = args.gate_temperature
    config.candidate_count = args.candidate_count
    config.budget_loss_weight = args.budget_loss_weight


def build_train_objects(args: argparse.Namespace) -> Tuple[CCVRL, CCVRLConfig, TextBatchCollator]:
    backbone = HuggingFaceBackbone(args.model_name)
    config = CCVRLConfig.for_task(args.task, hidden_size=backbone.hidden_size)
    apply_overrides(config, args)
    model = CCVRL(backbone, config)
    prototypes = load_prototype_tensor(args.prototypes)
    model.set_prototypes(prototypes)
    collator = TextBatchCollator(args.model_name, config.max_length, config.num_labels)
    return model, config, collator


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, collator = build_train_objects(args)
    train_set = MultiLabelJsonlDataset(args.train, config.num_labels)
    validation_set = MultiLabelJsonlDataset(args.validation, config.num_labels)
    test_set = MultiLabelJsonlDataset(args.test, config.num_labels)

    reversal_pairs = load_pair_tensor(args.reversal_pairs) if args.reversal_pairs else None
    stability_pairs = load_pair_tensor(args.stability_pairs) if args.stability_pairs else None
    train_loader = build_train_loader(
        train_set,
        collator,
        config.batch_size,
        args.workers,
        reversal_pairs,
        stability_pairs,
        config.seed,
    )
    validation_loader = build_loader(validation_set, collator, config.batch_size, False, args.workers)
    test_loader = build_loader(test_set, collator, config.batch_size, False, args.workers)

    trainer = CCVRLTrainer(
        model,
        config,
        device=args.device,
        use_amp=not args.no_amp,
        reversal_pairs=reversal_pairs,
        stability_pairs=stability_pairs,
    )
    fit_result = trainer.fit(
        train_loader,
        validation_loader,
        epochs=config.epochs,
        label_frequency=train_set.label_frequency(),
    )
    test_metrics = trainer.evaluate(
        test_loader,
        label_frequency=train_set.label_frequency(),
    )

    trainer.save_checkpoint(output_dir / "best_model.pt")
    trainer.save_metrics(fit_result, output_dir / "training_history.json")
    trainer.save_metrics(test_metrics, output_dir / "test_metrics.json")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    print(json.dumps(test_metrics, indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = CCVRLConfig(**checkpoint["config"])
    backbone = HuggingFaceBackbone(args.model_name)
    if backbone.hidden_size != config.hidden_size:
        raise ValueError("Checkpoint and backbone hidden sizes differ")
    model = CCVRL(backbone, config)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    collator = TextBatchCollator(args.model_name, config.max_length, config.num_labels)
    test_set = MultiLabelJsonlDataset(args.test, config.num_labels)
    test_loader = build_loader(test_set, collator, args.batch_size, False, args.workers)
    label_frequency = None
    if args.train is not None:
        train_set = MultiLabelJsonlDataset(args.train, config.num_labels)
        label_frequency = train_set.label_frequency()

    trainer = CCVRLTrainer(model, config, device=args.device, use_amp=not args.no_amp)
    metrics = trainer.evaluate(test_loader, label_frequency=label_frequency)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCVRL training and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate CCVRL")
    train_parser.add_argument("--task", choices=["corevalue", "c-varc-8"], required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--validation", required=True)
    train_parser.add_argument("--test", required=True)
    train_parser.add_argument("--prototypes", required=True)
    train_parser.add_argument("--reversal-pairs", default=None)
    train_parser.add_argument("--stability-pairs", default=None)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--no-amp", action="store_true")

    train_parser.add_argument("--tau-c", type=float, default=0.80)
    train_parser.add_argument("--tau-b", type=float, default=0.50)
    train_parser.add_argument("--prototype-temperature", type=float, default=0.07)
    train_parser.add_argument("--gate-temperature", type=float, default=0.10)
    train_parser.add_argument("--candidate-count", type=int, default=4)
    train_parser.add_argument("--budget-loss-weight", type=float, default=0.10)
    train_parser.set_defaults(func=train_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--test", required=True)
    evaluate_parser.add_argument("--train", default=None)
    evaluate_parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--workers", type=int, default=2)
    evaluate_parser.add_argument("--device", default=None)
    evaluate_parser.add_argument("--no-amp", action="store_true")
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
