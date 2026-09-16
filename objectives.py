from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from config import CCVRLConfig


PairTensor = torch.Tensor


def build_target_signature(
    labels: torch.Tensor,
    directions: Optional[torch.Tensor],
    task_name: str,
) -> torch.Tensor:
    """Construct category-only or signed category-direction targets."""

    if task_name == "corevalue":
        if directions is None:
            raise ValueError("CoreValue requires direction labels")
        signed_direction = torch.zeros_like(labels)
        active = (labels > 0.5) & (directions >= 0)
        signed_direction[active] = directions[active].to(labels.dtype) * 2.0 - 1.0
        return labels * signed_direction
    return labels


def _upper_triangle_pairs(mask: torch.Tensor) -> PairTensor:
    mask = torch.triu(mask, diagonal=1)
    return mask.nonzero(as_tuple=False)


@torch.no_grad()
def mine_reversal_pairs(
    action: torch.Tensor,
    target_signature: torch.Tensor,
    action_similarity_threshold: float,
    target_difference_threshold: float,
) -> PairTensor:
    """Select action-similar samples with sufficiently different targets."""

    normalized = F.normalize(action.detach(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    target_difference = torch.cdist(target_signature.float(), target_signature.float(), p=1)
    mask = (similarity >= action_similarity_threshold) & (
        target_difference >= target_difference_threshold
    )
    return _upper_triangle_pairs(mask)


@torch.no_grad()
def mine_stability_pairs(
    core_condition: torch.Tensor,
    weak_background: torch.Tensor,
    target_signature: torch.Tensor,
    core_similarity_threshold: float,
    background_similarity_threshold: float,
) -> PairTensor:
    """Select target-matched pairs with stable core conditions and distinct weak background."""

    core = F.normalize(core_condition.detach(), dim=-1)
    background = F.normalize(weak_background.detach(), dim=-1)
    core_similarity = core @ core.transpose(0, 1)
    background_similarity = background @ background.transpose(0, 1)
    target_equal = torch.cdist(target_signature.float(), target_signature.float(), p=1) == 0
    mask = (
        target_equal
        & (core_similarity >= core_similarity_threshold)
        & (background_similarity <= background_similarity_threshold)
    )
    return _upper_triangle_pairs(mask)


def reversal_margin_loss(
    relation_signature: torch.Tensor,
    pairs: PairTensor,
    margin: float,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return relation_signature.sum() * 0.0
    first = relation_signature[pairs[:, 0]]
    second = relation_signature[pairs[:, 1]]
    distance = torch.linalg.vector_norm(first - second, dim=-1)
    return F.relu(margin - distance).mean()


def bernoulli_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1.0 - 1e-7)
    q = q.clamp(1e-7, 1.0 - 1e-7)
    m = 0.5 * (p + q)

    def bernoulli_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * (a.log() - b.log()) + (1.0 - a) * (
            (1.0 - a).log() - (1.0 - b).log()
        )

    return 0.5 * (bernoulli_kl(p, m) + bernoulli_kl(q, m))


def stability_loss(light_probabilities: torch.Tensor, pairs: PairTensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return light_probabilities.sum() * 0.0
    first = light_probabilities[pairs[:, 0]]
    second = light_probabilities[pairs[:, 1]]
    return bernoulli_js_divergence(first, second).mean()


def direction_loss(direction_logits: torch.Tensor, directions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = (labels > 0.5) & (directions >= 0)
    if not valid.any():
        return direction_logits.sum() * 0.0
    return F.cross_entropy(direction_logits[valid], directions[valid])


def compute_ccvrl_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    directions: Optional[torch.Tensor],
    config: CCVRLConfig,
    reversal_pairs: Optional[PairTensor] = None,
    stability_pairs: Optional[PairTensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute prediction, relation, stability, and compute-budget objectives."""

    category = F.binary_cross_entropy_with_logits(outputs["category_logits"], labels)
    targets = build_target_signature(labels, directions, config.task_name)

    if reversal_pairs is None:
        reversal_pairs = mine_reversal_pairs(
            outputs["action"],
            targets,
            config.action_similarity_threshold,
            config.target_difference_threshold,
        )
    if stability_pairs is None:
        stability_pairs = mine_stability_pairs(
            outputs["core_condition"],
            outputs["weak_background"],
            targets,
            config.stability_core_threshold,
            config.stability_background_threshold,
        )

    reversal = reversal_margin_loss(
        outputs["relation_signature"], reversal_pairs, config.reversal_margin
    )
    stable = stability_loss(outputs["light_probabilities"], stability_pairs)

    if config.task_name == "corevalue":
        if directions is None:
            raise ValueError("CoreValue requires directions")
        direction = direction_loss(outputs["direction_logits"], directions, labels)
    else:
        direction = outputs["direction_logits"].sum() * 0.0

    budget = F.relu(outputs["average_cost"] - outputs["budget_target"]).pow(2)
    total = (
        category
        + config.direction_loss_weight * direction
        + config.reversal_loss_weight * reversal
        + config.stability_loss_weight * stable
        + config.budget_loss_weight * budget
    )

    return {
        "loss": total,
        "category_loss": category,
        "direction_loss": direction,
        "reversal_loss": reversal,
        "stability_loss": stable,
        "budget_loss": budget,
        "reversal_pairs": labels.new_tensor(float(reversal_pairs.size(0))),
        "stability_pairs": labels.new_tensor(float(stability_pairs.size(0))),
    }
