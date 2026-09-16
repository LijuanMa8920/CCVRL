from __future__ import annotations

from typing import Dict, Optional

import torch


def _safe_f1(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    denominator = 2.0 * tp + fp + fn
    return torch.where(denominator > 0, 2.0 * tp / denominator, torch.zeros_like(denominator))


def micro_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.bool()
    gold = target.bool()
    tp = (pred & gold).sum().float()
    fp = (pred & ~gold).sum().float()
    fn = (~pred & gold).sum().float()
    return float(_safe_f1(tp, fp, fn).item())


def macro_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.bool()
    gold = target.bool()
    tp = (pred & gold).sum(dim=0).float()
    fp = (pred & ~gold).sum(dim=0).float()
    fn = (~pred & gold).sum(dim=0).float()
    return float(_safe_f1(tp, fp, fn).mean().item())


def multi_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    subset = target.sum(dim=1) >= 2
    if not subset.any():
        return 0.0
    return micro_f1(prediction[subset], target[subset])


def direction_f1(
    direction_prediction: torch.Tensor,
    direction_target: torch.Tensor,
    category_target: torch.Tensor,
) -> float:
    valid = (category_target > 0.5) & (direction_target >= 0)
    if not valid.any():
        return 0.0

    pred = direction_prediction[valid]
    gold = direction_target[valid]
    scores = []
    for class_id in (0, 1):
        p = pred == class_id
        g = gold == class_id
        tp = (p & g).sum().float()
        fp = (p & ~g).sum().float()
        fn = (~p & g).sum().float()
        scores.append(_safe_f1(tp, fp, fn))
    return float(torch.stack(scores).mean().item())


def joint_f1(
    category_prediction: torch.Tensor,
    category_target: torch.Tensor,
    direction_prediction: torch.Tensor,
    direction_target: torch.Tensor,
) -> float:
    pred_category = category_prediction.bool()
    gold_category = category_target.bool()
    direction_known = direction_target >= 0
    direction_correct = direction_prediction == direction_target

    tp_mask = pred_category & gold_category & direction_known & direction_correct
    fp_mask = pred_category & (
        ~gold_category | (gold_category & direction_known & ~direction_correct)
    )
    fn_mask = gold_category & (
        ~pred_category | (pred_category & direction_known & ~direction_correct)
    )
    tp = tp_mask.sum().float()
    fp = fp_mask.sum().float()
    fn = fn_mask.sum().float()
    return float(_safe_f1(tp, fp, fn).item())


def expected_calibration_error(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    bins: int = 15,
) -> float:
    probs = probabilities.detach().float().reshape(-1).clamp(0.0, 1.0)
    gold = target.detach().float().reshape(-1)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    ece = probs.new_tensor(0.0)
    total = probs.numel()
    for idx in range(bins):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if mask.any():
            confidence = probs[mask].mean()
            accuracy = gold[mask].mean()
            ece += mask.float().sum() / total * (confidence - accuracy).abs()
    return float(ece.item())


def tail_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_frequency: torch.Tensor,
    fraction: float = 0.25,
) -> float:
    num_labels = target.size(1)
    tail_count = max(1, int(round(num_labels * fraction)))
    tail_ids = torch.argsort(label_frequency)[:tail_count]
    return macro_f1(prediction[:, tail_ids], target[:, tail_ids])


def difficulty_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ambiguity: torch.Tensor,
    fraction: float = 0.30,
) -> Dict[str, float]:
    sample_count = target.size(0)
    subset_size = max(1, int(round(sample_count * fraction)))
    order = torch.argsort(ambiguity)
    easy_ids = order[:subset_size]
    hard_ids = order[-subset_size:]
    return {
        "easy_f1": micro_f1(prediction[easy_ids], target[easy_ids]),
        "hard_f1": micro_f1(prediction[hard_ids], target[hard_ids]),
    }


def relation_pair_accuracy(
    category_prediction: torch.Tensor,
    direction_prediction: Optional[torch.Tensor],
    category_target: torch.Tensor,
    direction_target: Optional[torch.Tensor],
    pairs: torch.Tensor,
    task_name: str,
) -> float:
    if pairs.numel() == 0:
        return 0.0

    if task_name == "corevalue":
        if direction_prediction is None or direction_target is None:
            raise ValueError("CoreValue pair accuracy requires direction tensors")
        signed_gold = category_target * (direction_target.clamp_min(0).float() * 2.0 - 1.0)
        signed_pred = category_prediction.float() * (direction_prediction.float() * 2.0 - 1.0)
        changed = signed_gold[pairs[:, 0]] != signed_gold[pairs[:, 1]]
        first_ok = signed_pred[pairs[:, 0]] == signed_gold[pairs[:, 0]]
        second_ok = signed_pred[pairs[:, 1]] == signed_gold[pairs[:, 1]]
        pair_ok = ((first_ok & second_ok) | ~changed).all(dim=1)
    else:
        first_ok = category_prediction[pairs[:, 0]] == category_target[pairs[:, 0]].bool()
        second_ok = category_prediction[pairs[:, 1]] == category_target[pairs[:, 1]].bool()
        pair_ok = (first_ok & second_ok).all(dim=1)
    return float(pair_ok.float().mean().item())


def evaluate_predictions(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    ambiguity: torch.Tensor,
    threshold: float = 0.5,
    direction_logits: Optional[torch.Tensor] = None,
    directions: Optional[torch.Tensor] = None,
    label_frequency: Optional[torch.Tensor] = None,
    task_name: str = "corevalue",
) -> Dict[str, float]:
    category_prediction = probabilities >= threshold
    result = {
        "macro_f1": macro_f1(category_prediction, labels),
        "micro_f1": micro_f1(category_prediction, labels),
        "multi_f1": multi_f1(category_prediction, labels),
        "ece": expected_calibration_error(probabilities, labels, bins=15),
    }
    result.update(difficulty_f1(category_prediction, labels, ambiguity, fraction=0.30))

    if label_frequency is not None:
        result["tail_f1"] = tail_f1(category_prediction, labels, label_frequency)

    if task_name == "corevalue" and direction_logits is not None and directions is not None:
        direction_prediction = direction_logits.argmax(dim=-1)
        result["direction_f1"] = direction_f1(direction_prediction, directions, labels)
        result["joint_f1"] = joint_f1(
            category_prediction,
            labels,
            direction_prediction,
            directions,
        )
    return result
