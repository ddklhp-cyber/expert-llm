# expert-llm

> A modular LLM architecture with domain-specialized expert FFNs and selective gradient routing.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [Benchmark](#benchmark)
- [Generation Examples](#generation-examples)
- [Comparison with Related Work](#comparison-with-related-work)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

**expert-llm** is a transformer where a shared attention + base FFN learns general patterns, while multiple small expert FFNs learn domain-specific knowledge. All components train jointly — domain labels route gradients to the correct expert.

```
Input: "2 + 3 = 5"  (domain: math)

┌───────────────────────────────────────────┐
│ Shared Attention (always active)          │
│ Shared FFN_base (always active)           │
│ Expert FFN_math (active for this input)   │
│ Expert FFN_language (inactive)            │
│ Expert FFN_code (inactive)                │
└───────────────────────────────────────────┘

Forward:
  hidden = attention(input)
  hidden = hidden + FFN_base(hidden) + FFN_math(hidden)
```

### Key Features

- **Selective gradient routing** — shared layers train on all data, experts only on their domain
- **Modular inference** — activate one or more experts at runtime
- **Additive expert composition** — combine multiple experts for cross-domain queries
- **Self-contained** — generates synthetic multi-domain data, no external datasets needed

---

## Quick Start

### Installation

Requires Python 3.8+ and PyTorch 2.0+.

```bash
git clone <repo-url>
cd expert-llm
```

### Training

Self-contained (no external data needed):

```bash
python training/train_expert_synthetic.py
```

With TinyStories data (requires [TinyStories-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories)):

```bash
python training/train_expert.py --data-path ./data/TinyStories-train.txt
```

### Inference

```bash
# Synthetic model (6 domains)
python generate.py --checkpoint checkpoints/expert_synthetic.pt --prompt "water boils"

# TinyStories model (4 domains)
python generate.py --checkpoint checkpoints/expert_model.pt --prompt "the cat"

# Specific expert(s)
python generate.py --checkpoint checkpoints/expert_synthetic.pt --experts 1 3 --prompt "3 plus 4"
```

Or use in code:

```python
import torch
from model import ExpertLLM

model = ExpertLLM(vocab_size=496, num_experts=6)
model.load_state_dict(torch.load('checkpoints/expert_synthetic.pt'))

output = model(input_ids, active_experts=[1])      # math expert
output = model(input_ids, active_experts=[1, 5])   # math + code
output = model(input_ids, active_experts=None)     # base only
```

---

## Architecture

### Tokenization

Word-level tokenization — each token is a whole word (not characters or subword pieces):

```python
"the cat sat on the mat" → ["the", "cat", "sat", "on", "the", "mat"] → [42, 7, 103, 15, 42, 89]
```

- Regex split: `[a-zA-Z']+`, lowercased
- Special tokens: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`
- Out-of-vocabulary words are dropped
- No BPE/SentencePiece — vocabulary is small and controlled

### Embedding

Word → vector lookup, weight-tied with output head:

```
input word IDs → nn.Embedding(vocab, 64) → hidden vectors
output: hidden → Linear(64, vocab) → logits (shares embedding weights)
```

Trained from random init. No pre-trained vectors (Word2Vec/GloVe). The LM head reuses embedding weights (weight tying) to reduce parameters.

### Attention

Multi-head self-attention with RoPE positional encoding:

```
Q, K, V = Linear(hidden) split into 8 heads × 8 dim each
Positions encoded via Rotary Position Embedding (RoPE)
Scores = softmax(Q·Kᵀ / √head_dim + causal_mask)
Output = Scores · V → Linear → hidden
```

Causal mask ensures each token only attends to previous tokens (autoregressive) — when predicting the next word, the model cannot "cheat" by looking ahead:

```
         the  cat  sat
the     [ ✓    ✗    ✗ ]   ← "the" can only see itself
cat     [ ✓    ✓    ✗ ]   ← "cat" sees "the" and itself
sat     [ ✓    ✓    ✓ ]   ← "sat" sees everything before it
```

RoPE (Rotary Position Embedding) encodes word order by rotating Q/K vectors based on position. Tokens close together get similar rotations, far apart get different ones. This encodes *relative* distance rather than absolute position — generalizes to unseen sequence lengths with no extra parameters. Used by LLaMA, Mistral, and most modern LLMs.

### FFN (SwiGLU)

Gated feed-forward network:

```
gate = Linear(hidden → ffn_size)
up   = Linear(hidden → ffn_size)
down = Linear(ffn_size → hidden)
output = down(SiLU(gate) * up)
```

SwiGLU gating (used in LLaMA/Mistral) — the gate controls information flow, more expressive than standard ReLU FFN.

### RMSNorm

Pre-norm before attention and FFN:

```
output = weight * x / RMS(x)
```

Traditional LayerNorm does `(x - mean) / std * weight + bias` (4 ops). RMSNorm drops the mean subtraction and bias (2 ops) — fewer computations, faster, and works just as well for LLMs in practice. Used by LLaMA, Mistral, and most modern architectures.

### Per-Layer Forward Pass

```
hidden = hidden + Attention(RMSNorm(hidden))        # residual + attention
hidden = hidden + FFN_base(RMSNorm(hidden))         # residual + base FFN
                + FFN_expert(RMSNorm(hidden))       # + expert FFN (if active)
```

### Model Summary

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

### Domains

| ID | Domain | Description |
|----|--------|-------------|
| 0 | reading | stories, vocabulary, simple sentences |
| 1 | math | arithmetic, numbers, word problems |
| 2 | nature | animals, weather, plants, seasons |
| 3 | science | physics, chemistry, biology facts |
| 4 | history | events, people, cause/effect |
| 5 | code | programming patterns, logic |

---

## Training

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

### Loss Computation

Each domain group computes its own cross-entropy loss independently, then gradients are accumulated before a single optimizer step:

```python
for domain_id in unique_domains_in_batch:
    # Select only samples belonging to this domain
    domain_samples = batch[domain == domain_id]

    # Forward with ONLY this domain's expert active
    logits = model(domain_samples, active_experts=[domain_id])

    # Standard next-token prediction loss
    loss = cross_entropy(logits, targets, ignore_index=PAD)

    # Scale by proportion of batch (so total gradient magnitude is consistent)
    scaled_loss = loss * (num_domain_samples / batch_size)

    # Backward — gradients accumulate in shared params,
    # but ONLY this expert's FFN gets gradients (others weren't in forward pass)
    scaled_loss.backward()

# Single optimizer step after all domains
optimizer.step()
```

**Why this works:**

- Each expert FFN only participates in forward/backward for its own domain → it never receives gradients from other domains
- Shared layers (attention, base FFN, embedding) participate in ALL forward passes → they receive gradients from every domain
- The `loss * (num_domain_samples / batch_size)` scaling ensures each sample contributes equally regardless of domain distribution in the batch
- No auxiliary losses, no load balancing — just standard cross-entropy with selective routing

### Expert Composition at Inference

Experts are additive in hidden space — activating multiple experts sums their contributions:

```python
# hidden = hidden + FFN_base(x) + FFN_expert_1(x) + FFN_expert_5(x)
output = model(input_ids, active_experts=[1, 5])
```

---

## Benchmark

### Expert vs Single Model

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

---

## Generation Examples

### Synthetic Model (`expert_synthetic.pt`)

**Domain-matched prompts:**

| Expert | Prompt | Output |
|--------|--------|--------|
| reading | "once upon a time" | "once upon a time there was a happy bird named anna" |
| math | "3 plus 4" | "3 plus 4 equals 7" |
| nature | "the bird" | "the bird is a reptile that can fly" |
| science | "water boils" | "water boils at 100 degrees and freezes at 0 degrees" |
| history | "long ago people" | "long ago people lived in villages and hunted owl for food" |
| code | "a loop" | "a loop repeats the same action 8 times" |

**Base only (no expert) — same prompts:**

| Prompt | Output |
|--------|--------|
| "once upon a time" | "once upon a time time time time time in the river" |
| "3 plus 4" | "3 plus 4 and the bear share share share" |
| "water boils" | "water boils key key key key key key key key..." |
| "a loop" | "a loop key key key key key key key key" |

Without experts, the model degenerates into repetition — experts provide the specialization needed for coherent generation.

**Wrong expert (cross-domain mismatch):**

| Setup | Output |
|-------|--------|
| math prompt + reading expert | "3 plus 4 bear became best friends" |
| reading prompt + math expert | "once upon a time 8 more now he has 16 star equally among 2 friends each got 4 apple" |
| nature prompt + code expert | "the bird times" |

**Multiple experts combined:**

| Experts | Prompt | Output |
|---------|--------|--------|
| nature + science | "the bird" | "the bird has thick fur to help it swim" |
| math + code | "3 plus 4" | "3 plus 4 equals 7" |
| history + nature | "long ago" | "long ago people lived mice warmth share knowledge" |

### TinyStories Model (`expert_model.pt`)

Prompt: "the cat..."

| Expert | Output | Character |
|--------|--------|-----------|
| narrative | "was very proud of himself and they had a lot of fun together" | Descriptive storytelling |
| dialogue | "said i want to go to the park" | Direct speech |
| action | "flew down and landed on a branch" | Physical movement |
| emotion | "was so happy that he had found the new toy and he was very proud" | Feelings |
| base only | "was so happy that the little girl had was so much fun" | Generic, weaker grammar |
| narrative + emotion | "proud of her special game, so proud" | Blended style |

---

## Comparison with Related Work

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

## Project Structure

```
expert-llm/
├── model/
│   ├── __init__.py                # Package exports
│   └── expert_transformer.py      # Model architecture (RMSNorm, RoPE, Attention, FFN, ExpertLayer, ExpertLLM)
├── training/
│   ├── train_expert_synthetic.py  # Self-contained training (synthetic data, 6 domains)
│   └── train_expert.py            # TinyStories training (requires external data, 4 domains)
├── generate.py                    # Inference / demo script
├── checkpoints/
│   ├── expert_synthetic.pt        # 6-domain model (vocab=496, 720K params)
│   ├── expert_model.pt            # 4-domain TinyStories model (vocab=5007, 862K params)
│   ├── vocab_synthetic.json       # Vocab for synthetic model
│   └── vocab_tinystories.json     # Vocab for TinyStories model
├── .gitignore
└── README.md
```

### Included Checkpoints

| File | Domains | Vocab | Params | Val Loss | Data |
|------|---------|-------|--------|----------|------|
| `expert_synthetic.pt` | 6 (reading, math, nature, science, history, code) | 496 | 720K | 0.629 | Synthetic (self-contained) |
| `expert_model.pt` | 4 (narrative, dialogue, action, emotion) | 5007 | 862K | 3.639 | TinyStories (external) |

---

## License

MIT
