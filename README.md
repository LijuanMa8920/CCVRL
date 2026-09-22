# CCVRL

## Efficient Context-Conditional Value Relation Learning with Adaptive Inference for Multi-Label Text Classification

This repository contains the research code and reproducibility materials for **Context-Conditional Value Relation Learning (CCVRL)**, a framework for multi-label value recognition in Chinese text. CCVRL separates action evidence from contextual conditions and selectively activates deeper relation reasoning only for ambiguous samples. The implementation follows the manuscript formulation of **Contextual Evidence Decomposition (CED)**, **Normative Value Multi-Prototype Matching (NVP)**, **Natural Value Semantic Reversal (NVR)**, **Background Stability Constraint (BSC)**, and **Ambiguity-driven Budgeted Relation Reasoning (ABRR)**.

Repository: https://github.com/LijuanMa8920/CCVRL.git  
Data and split information: https://doi.org/10.6084/m9.figshare.33769588

---

## 1. Description

CCVRL is designed for cases in which similar actions may correspond to different value labels because their objects, preconditions, or execution conditions differ. The model first organizes token representations into action, core-condition, and weak-background evidence. It then compares action and condition evidence independently with multiple value prototypes. During training, natural reversal pairs and background-stability pairs provide relation supervision. During inference, an ambiguity gate combines action-condition disagreement with predictive uncertainty and activates candidate-conditioned relation reasoning only when additional computation is needed.

The released implementation supports two experimental tasks:

- **CoreValue**: eight-category multi-label value recognition with activation/inhibition direction prediction at active value positions.
- **C-VARC-8**: eight-category multi-label value recognition derived from the C-VARC rule corpus using the value categories shared with CoreValue.

---

## 2. Dataset Information

### 2.1 Data availability

The dataset information, split metadata, and study-related data materials are available at:

https://doi.org/10.6084/m9.figshare.33769588

To reproduce the reported results, use the released fixed splits and do not construct new train/validation/test partitions.

### 2.2 CoreValue

CoreValue contains **6,994 Chinese occupational-behaviour sentences** and **34,965 event elements**. The task uses eight value categories:

1. civility
2. justice
3. equality
4. rule of law
5. patriotism
6. dedication
7. integrity
8. friendliness

Among the 6,994 sentences, 1,128 contain at least two value labels. In addition to category labels, CoreValue provides activation/inhibition directions at value positions and seven event-role types used by CED: subject, action, precondition, object, manner, time, and location.

The public split used in the experiments is:

| Split | Number of samples |
|---|---:|
| Training | 5,595 |
| Validation | 699 |
| Test | 700 |
| Total | 6,994 |

### 2.3 C-VARC-8

C-VARC is a Chinese value-rule corpus organized around 12 core values and 50 derived values. The study constructs **C-VARC-8** by retaining the eight categories shared with CoreValue: civility, justice, equality, rule of law, patriotism, dedication, integrity, and friendliness.

The preprocessing used for C-VARC-8 consists of:

1. Unicode NFKC normalization;
2. whitespace normalization;
3. merging records with identical normalized text;
4. combining labels of duplicate records by set union; and
5. applying the fixed multi-label stratified train/validation/test split released with the study.

C-VARC-8 provides category labels only; direction labels are not required for this task.

### 2.4 Input format

Each split is stored as JSON Lines (JSONL), with one sample per line.

#### CoreValue example

```json
{"text":"...","labels":[0,4],"directions":{"0":1,"4":-1}}
```

`labels` may be represented either as active label indices or as an eight-dimensional multi-hot vector.

`directions` may use either:

- `-1 / +1` for inhibition / activation; or
- `0 / 1` for inhibition / activation.

The data loader converts both conventions to the internal `0 / 1` representation and applies direction supervision only at active category positions.

#### C-VARC-8 example

```json
{"text":"...","labels":[1,5]}
```

---

## 3. Code Information

### 3.1 Project layout

```text
ccvrl_release/
├── config.py       # Central configuration and task-specific settings
├── data.py         # JSONL loading, tokenization, pair-aware batching, prototype clustering
├── model.py        # CED, NVP, ABRR, relation reasoning, category/direction prediction
├── objectives.py   # Category, direction, NVR, BSC, and compute-budget objectives
├── metrics.py      # Classification, direction, calibration, tail, and difficulty metrics
├── trainer.py      # Optimization, validation, hard-routing evaluation, checkpoint handling
├── run.py          # Command-line training and evaluation entry point
└── README.md
```

### 3.2 Method-to-code mapping

| Manuscript component | Main implementation | Core operation |
|---|---|---|
| CED | `ContextualEvidenceDecomposition` | Soft token-to-role assignment and role-weighted evidence aggregation |
| NVP | `NormativeValuePrototypeMatcher` | Multi-prototype cosine matching for action and condition evidence |
| NVR | `mine_reversal_pairs`, `reversal_margin_loss` | Separation of action-similar but target-different sample pairs |
| BSC | `mine_stability_pairs`, `stability_loss` | Bernoulli Jensen-Shannon consistency on stable pairs |
| ABRR | `CCVRL`, `CandidateConditionedRelationReasoner` | Ambiguity gating, Top-M candidate selection, iterative relation updates |
| Joint prediction | `CCVRL.direction_head` | Category logits and masked direction prediction |
| Compute budget | `compute_ccvrl_loss` | One-sided penalty on mean routing cost |

---

## 4. Requirements

Recommended software environment:

```text
Python >= 3.10
PyTorch >= 2.1
Transformers >= 4.40
NumPy >= 1.24
```

Install the core dependencies with:

```bash
pip install torch transformers numpy
```

The manuscript-scale experiments use the Chinese RoBERTa-base-scale encoder:

```text
hfl/chinese-roberta-wwm-ext
```

For strict reproduction, use the same encoder checkpoint, maximum sequence length, mixed-precision policy, batch size, and hardware configuration as the reported experiment.

The deployment measurements in the manuscript use one NVIDIA RTX 4090, FP16 inference, batch size 1, and a maximum input length of 256 tokens.

---

## 5. Installation and Usage Instructions

### 5.1 Clone the repository

```bash
git clone https://github.com/LijuanMa8920/CCVRL.git
cd CCVRL
```

### 5.2 Install dependencies

```bash
pip install torch transformers numpy
```

### 5.3 Prepare the dataset files

Place the fixed split files in the expected data directory. The training commands below assume names such as:

```text
data/corevalue_train.jsonl
data/corevalue_val.jsonl
data/corevalue_test.jsonl

data/cvarc8_train.jsonl
data/cvarc8_val.jsonl
data/cvarc8_test.jsonl
```

Use the released split metadata from the study data package rather than re-splitting the datasets.

### 5.4 Prepare the prototype bank

NVP expects a tensor with shape:

```text
[num_labels, prototypes_per_label, hidden_size]
```

The manuscript configurations use:

```text
CoreValue : [8, 6, H]
C-VARC-8  : [8, 8, H]
```

Save the tensor directly with `torch.save(...)`, or store it under the key `prototypes` in a dictionary. The helper `build_prototype_bank(...)` in `data.py` performs spherical clustering from pre-encoded rule embeddings and multi-hot rule labels.

Prototype construction must use training-derived resources only. Offline rule encoding and clustering are preprocessing operations and are excluded from online inference latency.

### 5.5 Prepare cached pair files

For manuscript-level training, NVR and BSC use cached global pair indices derived from frozen training-split representations. Store them as tensors with shape `[N, 2]`:

```python
reversal_pairs = torch.tensor([[12, 81], [35, 109], ...], dtype=torch.long)
stability_pairs = torch.tensor([[4, 63], [28, 91], ...], dtype=torch.long)

torch.save(reversal_pairs, "reversal_pairs.pt")
torch.save(stability_pairs, "stability_pairs.pt")
```

When these files are supplied, `PairAwareBatchSampler` keeps as many cached pair endpoints as possible in the same mini-batch while covering each training sample once per epoch.

If the pair files are omitted, the objective falls back to detached in-batch pair mining. This fallback is suitable for implementation checks, whereas the cached global pairs correspond more closely to the manuscript protocol.

### 5.6 Train on CoreValue

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

### 5.7 Train on C-VARC-8

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

Each training run stores:

```text
best_model.pt
config.json
training_history.json
test_metrics.json
```

### 5.8 Evaluate a trained model

```bash
python run.py evaluate \
  --checkpoint runs/corevalue/best_model.pt \
  --test data/corevalue_test.jsonl \
  --train data/corevalue_train.jsonl \
  --model-name hfl/chinese-roberta-wwm-ext
```

Supplying the training split enables Tail-F1 computation from training-label frequencies.

---

## 6. Methodology

The implementation follows five main stages.

### 6.1 Contextual Evidence Decomposition (CED)

CED assigns soft semantic-role probabilities to token representations and aggregates them into action, core-condition, and weak-background evidence. The role inventory follows the event structure available in CoreValue.

### 6.2 Normative Value Multi-Prototype Matching (NVP)

Each value category is represented by multiple prototypes derived from training-side normative rules. Action and condition representations are matched independently against these prototypes so that the model retains separate action-based and condition-based support scores.

### 6.3 Natural Value Semantic Reversal (NVR)

NVR identifies training pairs with similar action representations but different value-category or direction targets. A margin-based relation objective separates these action-similar but target-different cases.

### 6.4 Background Stability Constraint (BSC)

BSC identifies target-matched samples with similar core conditions but different weak background information. It regularizes the light-path predictions so that irrelevant background variation does not unnecessarily alter the output.

### 6.5 Ambiguity-driven Budgeted Relation Reasoning (ABRR)

ABRR combines action-condition disagreement with predictive uncertainty. Samples below the routing threshold remain on the light path. Triggered samples receive candidate-conditioned relation updates for a fixed number of rounds. During deployment evaluation, the deep branch is executed only for triggered samples, so the reported computation reflects the actual hard-routing path.

---

## 7. Manuscript Hyperparameters

The settings explicitly reported in the manuscript are encoded in `CCVRLConfig.for_task(...)` or its shared defaults.

| Parameter | CoreValue | C-VARC-8 |
|---|---:|---:|
| Prototype count `K` | 6 | 8 |
| Action threshold `tau_A` | 0.84 | 0.90 |
| Reversal margin `m_r` | 0.35 | 0.35 |
| Ambiguity weight `beta` | 0.40 | 0.40 |
| Routing threshold `tau_g` | 0.50 | 0.50 |
| Relation rounds `L_R` | 2 | 2 |
| Compute budget `rho_B` | 0.35 | 0.35 |
| `lambda_rev / lambda_inv` | 0.10 / 0.05 | 0.10 / 0.05 |

Other quantities defined by the method, including condition/background thresholds, prototype temperature, gate temperature, candidate count, direction-loss weight, and budget-loss weight, remain explicit configuration or command-line fields. For exact reproduction, set them to the values stored in the corresponding experimental configuration and log files.

---

## 8. Evaluation Metrics

`metrics.py` implements the following metrics:

- **Macro-F1**: equal-weight average of binary F1 across the eight value categories.
- **Micro-F1**: F1 computed from globally accumulated true positives, false positives, and false negatives.
- **Multi-F1**: F1 on samples containing at least two ground-truth value labels.
- **Direction-F1**: direction prediction at active CoreValue positions.
- **Joint-F1**: requires both the value category and its direction to be correct.
- **Tail-F1**: F1 on the least-frequent 25% of labels.
- **Expected Calibration Error (ECE)**: 15-bin calibration error.
- **Hard-F1 / Easy-F1**: performance on the top/bottom 30% ambiguity subsets.
- **Trigger Rate**: proportion of samples routed to the deep relation branch.

Metric functions return values in `[0, 1]`; multiply by 100 when preparing percentage tables.

---

## 9. Five-Run Reproduction Protocol

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

After all paired runs are complete, aggregate the mean, sample standard deviation, paired differences, confidence intervals, and corrected significance tests using the same evaluation protocol.

---

## 10. Reproducibility Notes

- Validation data control model and hyperparameter selection; test data are used only after the final validation checkpoint has been selected.
- Prototype tensors, rule encodings, and cached NVR/BSC pair indices must be derived from training data only.
- Hard routing is used for deployment evaluation, and the deep relation branch is executed only for triggered samples.
- Offline rule encoding and prototype clustering remain outside the online latency boundary.
- Keep maximum input length, precision mode, batch size, warm-up procedure, and hardware fixed when reproducing deployment measurements.
- The synthetic `TinyBackbone` in `model.py` is intended only for offline code checks. Manuscript experiments should use the intended pretrained Chinese encoder.

---

## 11. Code Verification

Before release, the implementation was checked in three stages:

1. **Architecture and gradient verification**: syntax compilation, CoreValue/C-VARC-8 forward paths, finite losses, backward propagation, output dimensions, and soft/hard routing interfaces.
2. **Boundary-condition verification**: padded inputs, batch size one, no-trigger/all-trigger execution, controlled NVR/BSC pair selection, perfect-prediction metric checks, and direction-label encoding.
3. **Training-path verification**: one-epoch optimization, hard-routing evaluation, finite metrics, cached-pair localization, and pair-aware mini-batch preservation.

A final release check confirmed that cached global NVR pairs can be preserved within training batches and contribute to the relation-loss path.

---

## 12. Citation

If you use this repository, please cite the accompanying manuscript:

> Lijuan Ma and Congna He. **Efficient Context-Conditional Value Relation Learning with Adaptive Inference for Multi-Label Text Classification.** Accompanying manuscript for the CCVRL repository.

For the datasets, please also cite their original sources where applicable.

### CoreValue

> Liu, P., Zhang, S., Yu, D., and Bo, L. (2022). **CoreValue: Chinese Core Value-Behavior Frame and Knowledge Base for Value Computing.** Proceedings of the 21st Chinese National Conference on Computational Linguistics, 417-430.

https://aclanthology.org/2022.ccl-1.38/

### C-VARC

> Wu, P., Shen, G., Zhao, D., Wang, Y., Dong, Y., Shi, Y., Lu, E., Zhao, F., and Zeng, Y. (2025). **C-VARC: A Large-Scale Chinese Value Rule Corpus for Value Alignment of Large Language Models.** arXiv:2506.01495.

https://arxiv.org/abs/2506.01495

Study data and split information:

https://doi.org/10.6084/m9.figshare.33769588

---

## 13. License and Contribution Guidelines

### License

No separate software license is specified in this README. If a `LICENSE` file is included in the repository, that file governs use and redistribution of the code. Dataset use remains subject to the terms of the corresponding original or repository sources.

### Contributions and questions

For reproducibility questions, bug reports, or implementation issues, please use the GitHub issue tracker:

https://github.com/LijuanMa8920/CCVRL/issues

When reporting an issue, include the task name, software environment, command used, random seed, and the relevant error message or log excerpt. Proposed code changes should preserve the released data splits and evaluation protocol when they are intended to reproduce the manuscript results.

---

## 14. Contact

For questions related to the manuscript or repository, contact:

**Lijuan Ma**  
School of Intelligent Control, Changzhou Vocational Institute of Industry Technology  
Email: 13921098920@163.com

