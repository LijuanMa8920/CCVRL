# CCVRL 🧭
## Context-Conditional Value Relation Learning with Adaptive Inference for Multi-Label Text Classification

> **Research code for context-sensitive value recognition with selective post-encoder reasoning.**  
> The implementation follows the manuscript formulation of **CED + NVP + NVR + BSC + ABRR**, with separate category and direction heads for CoreValue and category prediction for C-VARC-8.

---

## ✦ Highlights

- **Contextual Evidence Decomposition (CED)** — softly organizes token features into action, core-condition, and weak-background evidence over seven semantic roles.
- **Normative Value Multi-Prototype Matching (NVP)** — compares action and condition evidence independently against multiple label prototypes.
- **Natural Value Semantic Reversal (NVR)** — separates action-similar samples whose category or polarity targets differ.
- **Background Stability Constraint (BSC)** — regularizes light-path predictions for target-matched samples with similar core conditions but different weak background.
- **Ambiguity-driven Budgeted Relation Reasoning (ABRR)** — combines action-condition disagreement and predictive uncertainty, then activates candidate-conditioned relation updates only when required.
- **Deployment-aware execution** — hard inference computes the deep branch only for triggered samples; untriggered samples remain on the light path.
- **Pair-aware training** — optional cached pair files preserve global NVR/BSC relations inside mini-batches through a dedicated batch sampler.

---

## 📁 Project Layout

```text
ccvrl_release/
├── config.py       # Central configuration and task-specific settings
├── data.py         # JSONL dataset, tokenizer collator, pair-aware batching, prototype clustering
├── model.py        # CED, NVP, ABRR, relation reasoning, category/direction prediction
├── objectives.py   # Category, direction, NVR, BSC, and compute-budget objectives
├── metrics.py      # Multi-label, direction, joint, calibration, tail, and difficulty metrics
├── trainer.py      # Optimization, validation, hard-routing evaluation, checkpoint handling
├── run.py          # Command-line training and evaluation entry point
└── README.md
```

---

## 🧠 Method-to-Code Mapping

| Manuscript component | Main implementation | Core operation |
|---|---|---|
| CED | `ContextualEvidenceDecomposition` | Soft token-to-role assignment and role-weighted evidence aggregation |
| NVP | `NormativeValuePrototypeMatcher` | Multi-prototype cosine matching for action and condition evidence |
| NVR | `mine_reversal_pairs`, `reversal_margin_loss` | Action-similar / target-different pair separation |
| BSC | `mine_stability_pairs`, `stability_loss` | Bernoulli Jensen-Shannon consistency on stable pairs |
| ABRR | `CCVRL`, `CandidateConditionedRelationReasoner` | Ambiguity gate, Top-M candidates, iterative condition updates |
| Joint prediction | `CCVRL.direction_head` | Category logits plus masked direction prediction |
| Compute budget | `compute_ccvrl_loss` | One-sided budget penalty on mean routing cost |

---

## ⚙️ Environment

Recommended environment:

```bash
python >= 3.10
pytorch >= 2.1
transformers >= 4.40
numpy >= 1.24
```

Install the core dependencies:

```bash
pip install torch transformers numpy
```

For the manuscript-scale encoder, the command-line interface defaults to:

```text
hfl/chinese-roberta-wwm-ext
```

Use the exact encoder checkpoint employed in the final experiment when reproducing reported results.

---

## 🗂️ Input Format

Each split is stored as JSONL. Every line contains one sample.

### CoreValue

```json
{"text":"...","labels":[0,4],"directions":{"0":1,"4":-1}}
```

`labels` may be either active label indices or an eight-dimensional multi-hot vector.

`directions` may use either:

- `-1 / +1` for inhibition / activation, or
- `0 / 1` for inhibition / activation.

The loader maps both conventions to the internal `0 / 1` representation and applies direction supervision only at active category positions.

### C-VARC-8

```json
{"text":"...","labels":[1,5]}
```

Direction labels are not required for C-VARC-8.

---

## 🧩 Prototype Bank

NVP expects a tensor with shape:

```text
[num_labels, prototypes_per_label, hidden_size]
```

For the two tasks:

```text
CoreValue : [8, 6, H]
C-VARC-8  : [8, 8, H]
```

Save the tensor directly with `torch.save(...)`, or store it under the key `prototypes` in a dictionary. The helper `build_prototype_bank(...)` in `data.py` performs spherical clustering from pre-encoded rule embeddings and multi-hot rule labels.

---

## 🔗 Cached Pair Files

The manuscript defines NVR and BSC pairs from frozen training-split representations. For the closest training behavior, store global pair indices as tensors of shape `[N, 2]`:

```python
reversal_pairs = torch.tensor([[12, 81], [35, 109], ...], dtype=torch.long)
stability_pairs = torch.tensor([[4, 63], [28, 91], ...], dtype=torch.long)

torch.save(reversal_pairs, "reversal_pairs.pt")
torch.save(stability_pairs, "stability_pairs.pt")
```

When these files are supplied, `PairAwareBatchSampler` keeps as many cached pair endpoints as possible in the same mini-batch while covering each training sample once per epoch.

If pair files are omitted, the training objective falls back to detached in-batch pair mining. That mode is useful for implementation checks, but global cached pairs are preferred for manuscript-level experiments.

---

## 🚀 Training

### CoreValue

```bash
python run.py train \
  --task corevalue \
  --train data/corevalue_train.jsonl \
  --validation data/corevalue_val.jsonl \
  --test data/corevalue_test.jsonl \
  --prototypes assets/corevalue_prototypes.pt \
  --reversal-pairs assets/corevalue_reversal_pairs.pt \
  --stability-pairs assets/corevalue_stability_pairs.pt \
  --output-dir runs/corevalue \
  --model-name hfl/chinese-roberta-wwm-ext \
  --batch-size 16 \
  --epochs 10 \
  --seed 42
```

### C-VARC-8

```bash
python run.py train \
  --task c-varc-8 \
  --train data/cvarc8_train.jsonl \
  --validation data/cvarc8_val.jsonl \
  --test data/cvarc8_test.jsonl \
  --prototypes assets/cvarc8_prototypes.pt \
  --reversal-pairs assets/cvarc8_reversal_pairs.pt \
  --stability-pairs assets/cvarc8_stability_pairs.pt \
  --output-dir runs/cvarc8 \
  --model-name hfl/chinese-roberta-wwm-ext \
  --batch-size 16 \
  --epochs 10 \
  --seed 42
```

Training stores:

```text
best_model.pt
config.json
training_history.json
test_metrics.json
```

---

## 🔍 Evaluation

```bash
python run.py evaluate \
  --checkpoint runs/corevalue/best_model.pt \
  --test data/corevalue_test.jsonl \
  --train data/corevalue_train.jsonl \
  --model-name hfl/chinese-roberta-wwm-ext
```

Supplying the training split enables Tail-F1 computation from training-label frequencies.

---

## 📊 Manuscript Hyperparameters

The settings explicitly reported in the manuscript are encoded directly in `CCVRLConfig.for_task(...)` or its shared defaults.

| Parameter | CoreValue | C-VARC-8 |
|---|---:|---:|
| Prototype count `K` | 6 | 8 |
| Action threshold `τ_A` | 0.84 | 0.90 |
| Reversal margin `m_r` | 0.35 | 0.35 |
| Ambiguity weight `β` | 0.40 | 0.40 |
| Routing threshold `τ_g` | 0.50 | 0.50 |
| Relation rounds `L_R` | 2 | 2 |
| Compute budget `ρ_B` | 0.35 | 0.35 |
| `λ_rev / λ_inv` | 0.10 / 0.05 | 0.10 / 0.05 |

Several quantities are defined by the method but are not numerically listed in the manuscript table, including `τ_C`, `τ_B`, prototype temperature, gate temperature, candidate count, direction-loss weight, and budget-loss weight. They remain explicit command-line or configuration fields so the released run can be aligned with the exact experimental log used for the final paper.

---

## 📐 Reported Metrics

`metrics.py` implements:

- Macro-F1
- Micro-F1
- Multi-F1 on samples with at least two positive labels
- Direction-F1 on active CoreValue positions
- Joint-F1 requiring category and direction agreement
- Tail-F1 on the least-frequent 25% of labels
- 15-bin Expected Calibration Error
- Hard-F1 / Easy-F1 from the top / bottom 30% ambiguity subsets
- Trigger rate for the hard deep branch

Metric functions return values in `[0, 1]`. Multiply by `100` for percentage tables.

---

## 🧪 Five-Run Protocol

The manuscript reports five paired runs. Keep the data split, prototype bank, pair cache, thresholds, and evaluation code fixed across seeds.

```bash
for seed in 11 23 37 53 71; do
  python run.py train \
    --task corevalue \
    --train data/corevalue_train.jsonl \
    --validation data/corevalue_val.jsonl \
    --test data/corevalue_test.jsonl \
    --prototypes assets/corevalue_prototypes.pt \
    --reversal-pairs assets/corevalue_reversal_pairs.pt \
    --stability-pairs assets/corevalue_stability_pairs.pt \
    --output-dir runs/corevalue_seed_${seed} \
    --seed ${seed}
done
```

Aggregate mean, sample standard deviation, paired differences, confidence intervals, and corrected significance tests only after all paired runs have completed.

---

## ✅ Three-Stage Code Verification

The package was checked in three successive passes before packaging:

1. **Architecture and gradient pass** — syntax compilation, CoreValue/C-VARC-8 forward paths, loss finiteness, backward propagation, output dimensions, and soft/hard routing interfaces.
2. **Boundary-condition pass** — padded inputs, batch size one, no-trigger/all-trigger execution, controlled NVR/BSC pair selection, perfect-prediction metric sanity, and direction-label encoding checks.
3. **Training-path pass** — one-epoch optimizer execution, hard-routing evaluation, finite metric outputs, cached-pair localization, and pair-aware mini-batch preservation.

A final release check confirmed that cached global NVR pairs are preserved inside training batches and contribute to the relation loss path.

---

## 🧾 Reproducibility Notes

- Validation controls model selection; test data are used only after the final validation checkpoint is selected.
- Prototype tensors and cached pair indices should be built from training data only.
- Hard routing is used during deployment evaluation; the deep relation branch is executed only for triggered samples.
- Offline rule encoding and clustering should remain outside the online latency boundary.
- Keep maximum input length, mixed-precision policy, batch size, warm-up protocol, and hardware fixed when reporting deployment measurements.
- The synthetic `TinyBackbone` in `model.py` exists only for offline code checks; manuscript experiments should use the intended pretrained Chinese encoder.

---

<p align="center"><b>CCVRL</b> · Context-sensitive evidence · Selective relation reasoning · Budget-aware inference</p>
