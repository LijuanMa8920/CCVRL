from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CCVRLConfig


class HuggingFaceBackbone(nn.Module):
    """Contextual text encoder backed by a Hugging Face transformer."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for HuggingFaceBackbone") from exc

        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs).last_hidden_state


class TinyBackbone(nn.Module):
    """Compact transformer used for offline checks and shape validation."""

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 64,
        max_length: int = 256,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del token_type_ids
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = ~attention_mask.bool()
        return self.encoder(hidden, src_key_padding_mask=padding_mask)


class ContextualEvidenceDecomposition(nn.Module):
    """Softly decompose token features into action, core-condition, and weak-background evidence."""

    SUBJECT = 0
    ACTION = 1
    PRECONDITION = 2
    OBJECT = 3
    MANNER = 4
    TIME = 5
    LOCATION = 6

    CORE_ROLE_IDS = (PRECONDITION, OBJECT, MANNER, SUBJECT)
    BACKGROUND_ROLE_IDS = (TIME, LOCATION)

    def __init__(self, hidden_size: int, num_roles: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        if num_roles != 7:
            raise ValueError("CCVRL uses seven semantic roles")
        self.role_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_roles),
        )
        self.core_weight = nn.Linear(hidden_size, 1, bias=False)
        self.background_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, token_features: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        role_logits = self.role_classifier(token_features)
        role_probabilities = role_logits.softmax(dim=-1)
        valid = attention_mask.to(token_features.dtype).unsqueeze(-1)
        weighted_probabilities = role_probabilities * valid

        denominator = weighted_probabilities.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        role_representations = torch.einsum(
            "btr,bth->brh", weighted_probabilities, token_features
        ) / denominator

        action = role_representations[:, self.ACTION]
        core_roles = role_representations[:, self.CORE_ROLE_IDS, :]
        background_roles = role_representations[:, self.BACKGROUND_ROLE_IDS, :]

        core_attention = self.core_weight(core_roles).squeeze(-1).softmax(dim=-1)
        background_attention = self.background_weight(background_roles).squeeze(-1).softmax(dim=-1)
        core_condition = torch.einsum("br,brh->bh", core_attention, core_roles)
        weak_background = torch.einsum("br,brh->bh", background_attention, background_roles)

        return {
            "role_probabilities": role_probabilities,
            "role_representations": role_representations,
            "core_role_representations": core_roles,
            "action": action,
            "core_condition": core_condition,
            "weak_background": weak_background,
        }


class NormativeValuePrototypeMatcher(nn.Module):
    """Match action and condition evidence against multiple prototypes per label."""

    def __init__(self, config: CCVRLConfig) -> None:
        super().__init__()
        self.num_labels = config.num_labels
        self.prototype_count = config.prototype_count
        self.hidden_size = config.hidden_size
        self.temperature = config.prototype_temperature
        self.action_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.condition_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.register_buffer(
            "prototypes",
            torch.zeros(config.num_labels, config.prototype_count, config.hidden_size),
        )
        self.register_buffer("prototype_ready", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        expected = (self.num_labels, self.prototype_count, self.hidden_size)
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"Prototype shape must be {expected}, received {tuple(prototypes.shape)}")
        normalized = F.normalize(prototypes.to(self.prototypes.device, self.prototypes.dtype), dim=-1)
        self.prototypes.copy_(normalized)
        self.prototype_ready.fill_(True)

    def forward(
        self, action: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(self.prototype_ready.item()):
            raise RuntimeError("Prototype bank is not initialized")

        bank = F.normalize(self.prototypes, dim=-1)
        action_query = F.normalize(self.action_projection(action), dim=-1)
        condition_query = F.normalize(self.condition_projection(condition), dim=-1)

        action_similarity = torch.einsum("bd,lkd->blk", action_query, bank)
        condition_similarity = torch.einsum("bd,lkd->blk", condition_query, bank)
        action_score, action_index = action_similarity.max(dim=-1)
        condition_score, condition_index = condition_similarity.max(dim=-1)
        action_score = action_score / self.temperature
        condition_score = condition_score / self.temperature

        combined_similarity = 0.5 * (action_similarity + condition_similarity)
        closest_index = combined_similarity.argmax(dim=-1)
        batch_size = action.size(0)
        expanded = bank.unsqueeze(0).expand(batch_size, -1, -1, -1)
        gather_index = closest_index.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.hidden_size
        )
        closest_prototypes = expanded.gather(dim=2, index=gather_index).squeeze(2)
        return action_score, condition_score, closest_prototypes, closest_index


class CandidateConditionedRelationReasoner(nn.Module):
    """Apply iterative condition updates only to high-probability label candidates."""

    def __init__(self, config: CCVRLConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.num_labels = config.num_labels
        self.candidate_count = min(config.candidate_count, config.num_labels)
        self.rounds = config.relation_rounds

        self.state_initializer = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.value_projection = nn.Parameter(
            torch.empty(config.num_labels, hidden, hidden)
        )
        nn.init.xavier_uniform_(self.value_projection)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.layer_norm = nn.LayerNorm(hidden)
        self.delta_head = nn.Linear(hidden, 1)

    def forward(
        self,
        light_probabilities: torch.Tensor,
        light_logits: torch.Tensor,
        action: torch.Tensor,
        core_roles: torch.Tensor,
        closest_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_labels = light_probabilities.shape
        candidate_ids = light_probabilities.topk(self.candidate_count, dim=-1).indices
        hidden = self.hidden_size

        candidate_proto = closest_prototypes.gather(
            1, candidate_ids.unsqueeze(-1).expand(-1, -1, hidden)
        )
        candidate_action = action.unsqueeze(1).expand(-1, self.candidate_count, -1)
        candidate_light_logits = light_logits.gather(1, candidate_ids).unsqueeze(-1)
        state = self.state_initializer(
            torch.cat([candidate_proto, candidate_action, candidate_light_logits], dim=-1)
        )

        keys = self.key_projection(core_roles)
        selected_projection = self.value_projection[candidate_ids]
        projected_roles = torch.einsum("bmde,bre->bmrd", selected_projection, core_roles)

        for _ in range(self.rounds):
            query = self.query_projection(state)
            attention_logits = torch.einsum("bmd,brd->bmr", query, keys) / math.sqrt(hidden)
            attention = attention_logits.softmax(dim=-1)
            context = torch.einsum("bmr,bmrd->bmd", attention, projected_roles)
            update = self.update_mlp(torch.cat([state, context], dim=-1))
            state = self.layer_norm(state + update)

        candidate_delta = self.delta_head(state).squeeze(-1)
        full_delta = light_probabilities.new_zeros(batch_size, num_labels)
        full_delta.scatter_(1, candidate_ids, candidate_delta)

        full_relation_state = action.new_zeros(batch_size, num_labels, hidden)
        full_relation_state.scatter_(
            1,
            candidate_ids.unsqueeze(-1).expand(-1, -1, hidden),
            state,
        )
        return full_delta, full_relation_state, candidate_ids


def _label_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    normalized = probabilities.clamp_min(1e-8)
    return normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )


def _multilabel_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    return entropy.mean(dim=-1)


class CCVRL(nn.Module):
    """Context-Conditional Value Relation Learning with adaptive inference."""

    def __init__(self, backbone: nn.Module, config: CCVRLConfig) -> None:
        super().__init__()
        if getattr(backbone, "hidden_size", None) != config.hidden_size:
            raise ValueError("Backbone hidden size and CCVRLConfig.hidden_size must match")

        self.backbone = backbone
        self.config = config
        hidden = config.hidden_size
        self.ced = ContextualEvidenceDecomposition(hidden, config.num_roles, config.dropout)
        self.prototype_matcher = NormativeValuePrototypeMatcher(config)
        self.light_mixture_logits = nn.Parameter(torch.zeros(config.num_labels, 2))
        self.reasoner = CandidateConditionedRelationReasoner(config)
        self.relation_signature_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 2),
        )

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        self.prototype_matcher.set_prototypes(prototypes)

    def _relation_signature(self, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [action, condition, action * condition, (action - condition).abs()],
            dim=-1,
        )
        return self.relation_signature_head(features)

    def _reason_all(
        self,
        light_probabilities: torch.Tensor,
        light_logits: torch.Tensor,
        action: torch.Tensor,
        core_roles: torch.Tensor,
        closest_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reasoner(
            light_probabilities,
            light_logits,
            action,
            core_roles,
            closest_prototypes,
        )

    def _reason_triggered(
        self,
        hard_gate: torch.Tensor,
        light_probabilities: torch.Tensor,
        light_logits: torch.Tensor,
        action: torch.Tensor,
        core_roles: torch.Tensor,
        closest_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_labels = light_probabilities.shape
        hidden = action.size(-1)
        full_delta = light_probabilities.new_zeros(batch_size, num_labels)
        full_relation_state = action.new_zeros(batch_size, num_labels, hidden)
        full_candidate_ids = torch.full(
            (batch_size, self.reasoner.candidate_count),
            -1,
            dtype=torch.long,
            device=action.device,
        )

        triggered = hard_gate.bool().nonzero(as_tuple=False).flatten()
        if triggered.numel() == 0:
            return full_delta, full_relation_state, full_candidate_ids

        delta, relation_state, candidate_ids = self.reasoner(
            light_probabilities[triggered],
            light_logits[triggered],
            action[triggered],
            core_roles[triggered],
            closest_prototypes[triggered],
        )
        full_delta[triggered] = delta
        full_relation_state[triggered] = relation_state
        full_candidate_ids[triggered] = candidate_ids
        return full_delta, full_relation_state, full_candidate_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        hard_routing: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if hard_routing is None:
            hard_routing = not self.training

        token_features = self.backbone(input_ids, attention_mask, token_type_ids)
        evidence = self.ced(token_features, attention_mask)
        action = evidence["action"]
        condition = evidence["core_condition"]
        background = evidence["weak_background"]

        action_score, condition_score, closest_prototypes, closest_index = self.prototype_matcher(
            action, condition
        )
        action_probabilities = torch.sigmoid(action_score)
        condition_probabilities = torch.sigmoid(condition_score)

        mixture = self.light_mixture_logits.softmax(dim=-1)
        light_logits = mixture[:, 0].unsqueeze(0) * action_score + mixture[:, 1].unsqueeze(0) * condition_score
        light_probabilities = torch.sigmoid(light_logits)

        action_distribution = _label_distribution(action_probabilities)
        condition_distribution = _label_distribution(condition_probabilities)
        disagreement = _js_divergence(action_distribution, condition_distribution)
        uncertainty = _multilabel_entropy(light_probabilities)
        ambiguity = disagreement + self.config.ambiguity_entropy_weight * uncertainty

        soft_gate = torch.sigmoid(
            (ambiguity - self.config.routing_threshold) / max(self.config.gate_temperature, 1e-6)
        )
        hard_gate = (ambiguity >= self.config.routing_threshold).to(light_probabilities.dtype)

        if hard_routing:
            deep_delta, relation_state, candidate_ids = self._reason_triggered(
                hard_gate,
                light_probabilities,
                light_logits,
                action,
                evidence["core_role_representations"],
                closest_prototypes,
            )
            execution_gate = hard_gate
        else:
            deep_delta, relation_state, candidate_ids = self._reason_all(
                light_probabilities,
                light_logits,
                action,
                evidence["core_role_representations"],
                closest_prototypes,
            )
            execution_gate = soft_gate

        category_logits = light_logits + execution_gate.unsqueeze(-1) * deep_delta
        category_probabilities = torch.sigmoid(category_logits)

        batch_size, num_labels = category_probabilities.shape
        action_expanded = action.unsqueeze(1).expand(-1, num_labels, -1)
        condition_expanded = condition.unsqueeze(1).expand(-1, num_labels, -1)
        direction_features = torch.cat(
            [action_expanded, condition_expanded, closest_prototypes, relation_state],
            dim=-1,
        )
        direction_logits = self.direction_head(direction_features)
        relation_signature = self._relation_signature(action, condition)

        average_cost = (
            self.config.light_path_cost
            + self.config.deep_path_incremental_cost * soft_gate.mean()
        )
        budget_target = (
            self.config.light_path_cost
            + self.config.compute_budget_ratio * self.config.deep_path_incremental_cost
        )

        output: Dict[str, torch.Tensor] = {
            "category_logits": category_logits,
            "category_probabilities": category_probabilities,
            "light_logits": light_logits,
            "light_probabilities": light_probabilities,
            "direction_logits": direction_logits,
            "action_probabilities": action_probabilities,
            "condition_probabilities": condition_probabilities,
            "action_score": action_score,
            "condition_score": condition_score,
            "action": action,
            "core_condition": condition,
            "weak_background": background,
            "relation_signature": relation_signature,
            "disagreement": disagreement,
            "uncertainty": uncertainty,
            "ambiguity": ambiguity,
            "soft_gate": soft_gate,
            "hard_gate": hard_gate,
            "candidate_ids": candidate_ids,
            "closest_prototype_ids": closest_index,
            "role_probabilities": evidence["role_probabilities"],
            "average_cost": average_cost,
            "budget_target": category_logits.new_tensor(budget_target),
        }
        return output
