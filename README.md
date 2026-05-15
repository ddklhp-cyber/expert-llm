# expert-llm

> A modular LLM architecture with domain-specialized expert FFNs and selective gradient routing.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**expert-llm** is a transformer where a shared attention + base FFN learns general patterns, while multiple small expert FFNs learn domain-specific knowledge. All components train jointly — domain labels route gradients to the correct expert.

```
Input: "2 + 3 = 5"  (domain: math)

  ┌─────────────────────────────────────────────┐
  │  Shared Attention (always active)            │
  │  Shared FFN_base (always active)             │
  │  Expert FFN_math (active for this input)     │
  │  Expert FFN_language (inactive)              │
  │  Expert FFN_code (inactive)                  │
  └─────────────────────────────────────────────┘

Forward:
  hidden = attention(input)
  hidden = hidden + FFN_base(hidden) + FFN_math(hidden)
```

## Key Features

- **Selective gradient routing** — shared layers train on all data, experts only on their domain
- **Modular inference** — activate one or more experts at runtime
- **Additive expert composition** — combine multiple experts for cross-domain queries
- **Self-contained** — generates synthetic multi-domain data, no external datasets needed

---

## Quick Start

### Training

```bash
python training/train_expert_synthetic.py
```

No external data needed — generates synthetic domain data internally.

### Inference

```python
import torch
from model import ExpertLLM

# Load 6-domain model (vocab=496, trained on synthetic data)
model = ExpertLLM(vocab_size=496, num_experts=6)
model.load_state_dict(torch.load('checkpoints/expert_synthetic.pt'))

# Single domain
output = model(input_ids, active_experts=[1])      # math expert

# Multiple domains
output = model(input_ids, active_experts=[1, 5])   # math + code

# Base only (no expert)
output = model(input_ids, active_experts=None)
```

---

## How It Works

### Selective Gradient Routing

```
Batch: [math_sample, reading_sample, code_sample, math_sample]

Step 1: Group by domain
  math_group    → [sample_0, sample_3]
  reading_group → [sample_1]
  code_group    → [sample_2]

Step 2: Forward each group with its expert
  math_group    → model(x, active_experts=[1])   # FFN_base + FFN_math
  reading_group → model(x, active_experts=[0])   # FFN_base + FFN_reading
  code_group    → model(x, active_experts=[5])   # FFN_base + FFN_code

Step 3: Backward
  Attention + FFN_base → gradients from ALL groups
  FFN_math             → gradients from math_group only
  FFN_reading          → gradients from reading_group only
  FFN_code             → gradients from code_group only
```

### Expert Composition at Inference

Experts are additive in hidden space — activating multiple experts sums their contributions:

```python
# hidden = hidden + FFN_base(x) + FFN_expert_1(x) + FFN_expert_5(x)
output = model(input_ids, active_experts=[1, 5])
```

---

## Benchmark: Expert vs Single Model

Same synthetic data, same steps (50K), comparable parameter count:

| Model | Params | Avg Val Loss | vs Baseline |
|-------|--------|-------------|-------------|
| **expert-llm (base + 6 experts)** | 720K | **0.641** | **−21%** 🏆 |
| Single FFN (ffn=256, no experts) | 714K | 0.814 | baseline |

### Per-Domain Results

| Domain | Expert | Single | Expert Wins By |
|--------|--------|--------|----------------|
| science | 0.294 | 0.476 | −38% |
| code | 0.363 | 0.529 | −31% |
| history | 0.548 | 0.726 | −25% |
| nature | 0.702 | 0.879 | −20% |
| reading | 0.913 | 1.089 | −16% |
| math | 1.027 | 1.183 | −13% |

Expert wins on **every domain**. Gradient isolation prevents cross-domain interference.

### Generation Samples (TinyStories model, prompt: "the cat...")

| Mode | Output | Character |
|------|--------|-----------|
| narrative expert | "was very proud of himself and they had a lot of fun together" | Storytelling |
| dialogue expert | "said i want to go to the park" | Direct speech |
| action expert | "flew down and landed on a branch" | Physical movement |
| emotion expert | "was so happy that he had found the new toy and he was very proud" | Feelings |
| base only | "was so happy that the little girl had was so much fun" | Generic, weaker grammar |

---

## Comparison with Related Architectures

### vs Mixture of Experts (MoE) — Switch Transformer, Mixtral

| | MoE | expert-llm |
|---|-----|-----------|
| **Routing** | Learned gating network per token | External domain label per sequence |
| **Training** | All experts see all data; router learns | Each expert only gets its domain's gradients |
| **Base FFN** | None — experts replace FFN | Shared base always active, experts add on top |
| **Load balancing** | Auxiliary loss required | Not needed — domains are explicit |
| **Interpretability** | Opaque (what did expert 3 learn?) | Clear (expert 3 = science) |

**Pros:** No router instability, no expert collapse, deterministic routing, shared base provides strong foundation.
**Cons:** Requires domain labels, per-sequence (not per-token) routing, cannot discover unexpected specializations.

### vs LoRA / Adapters

| | LoRA | expert-llm |
|---|------|-----------|
| **Training** | Pre-train base → fine-tune adapters separately | Joint from scratch — base and experts co-train |
| **Base model** | Frozen during adaptation | Trains on ALL data continuously |
| **Composition** | Ad-hoc merging (weight averaging) | Native additive composition |
| **Cross-domain learning** | Adapters never see each other's data | Shared layers learn cross-domain representations |

**Pros:** Joint training means shared layers benefit all domains, principled composition, no two-stage pipeline.
**Cons:** Cannot retrofit onto existing pre-trained models, requires all domain data upfront.

### vs Multi-task Learning (shared trunk)

| | Multi-task | expert-llm |
|---|-----------|-----------|
| **Interference** | Negative transfer in shared FFN | Expert FFNs are isolated per domain |
| **Capacity** | Fixed, shared across tasks | Scales with number of experts |
| **Output** | Separate heads per task | Single LM head — experts modify hidden states |

**Pros:** No negative transfer in expert params, single output interface, can add domains without retraining shared layers.
**Cons:** More total parameters, requires domain classification at inference.

### When to Use What

| Scenario | Best Choice |
|----------|-------------|
| Domain labels available, want interpretable specialization | ✅ **expert-llm** |
| Want to compose multiple domains at inference | ✅ **expert-llm** |
| Want automatic specialization discovery | MoE |
| Large pre-trained model, add tasks cheaply | LoRA |
| Per-token expert routing needed | MoE |

---

## Architecture

| Component | Size | Scope |
|-----------|------|-------|
| Embedding | vocab → 64 | shared |
| Attention | 8 heads × 8 dim | shared (per layer) |
| FFN_base | 64 → 128 → 64 | shared (per layer) |
| FFN_expert | 64 → 64 → 64 | per domain (per layer) |
| LM head | 64 → vocab (weight-tied) | shared |
| Layers | 6 | — |

### Parameter Budget

| Configuration | Params |
|---------------|--------|
| Base only (no experts) | 438K |
| Base + 1 expert | 512K |
| Base + 6 experts (full) | 720K |

## Domains

| ID | Domain | Description |
|----|--------|-------------|
| 0 | reading | stories, vocabulary, simple sentences |
| 1 | math | arithmetic, numbers, word problems |
| 2 | nature | animals, weather, plants, seasons |
| 3 | science | physics, chemistry, biology facts |
| 4 | history | events, people, cause/effect |
| 5 | code | programming patterns, logic |

---

## Project Structure

```
expert-llm/
├── model/
│   ├── __init__.py
│   └── expert_transformer.py    # Model architecture (RMSNorm, RoPE, Attention, FFN, ExpertLayer, ExpertLLM)
├── training/
│   └── train_expert_synthetic.py # Self-contained training with synthetic domain data
├── checkpoints/
│   └── expert_synthetic.pt       # 6-domain model (vocab=496, 720K params)
├── .gitignore
└── README.md
```

## Included Checkpoint

| File | Domains | Vocab | Params | Val Loss |
|------|---------|-------|--------|----------|
| `expert_synthetic.pt` | 6 (reading, math, nature, science, history, code) | 496 | 720K | 0.629 |

---

## License

MIT
