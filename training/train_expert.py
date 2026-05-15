"""
Train ExpertLLM jointly with domain labels on TinyStories data.
Each batch has (tokens, domain_id). Gradient flows through:
  - Attention + FFN_base: ALL examples
  - FFN_expert[i]: only examples with domain_id == i

Domains:
  0: narrative  (stories, descriptions)
  1: dialogue   (conversations, quotes)
  2: action     (physical actions, movement)
  3: emotion    (feelings, reactions)

Usage:
    python training/train_expert.py --data-path ./data/TinyStories-train.txt
"""
import sys
import time
import math
import random
import re
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.expert_transformer import ExpertLLM

# === Config ===
VOCAB_SIZE = 5007
HIDDEN = 64
NUM_LAYERS = 6
NUM_HEADS = 8
HEAD_DIM = 8
BASE_FFN = 128
EXPERT_FFN = 64
NUM_EXPERTS = 4
MAX_LEN = 32
BATCH_SIZE = 64
LR = 5e-4
STEPS = 50000
EVAL_INTERVAL = 2000

DOMAINS = {
    0: "narrative",
    1: "dialogue",
    2: "action",
    3: "emotion",
}


def classify_sentence(sentence):
    """Simple rule-based domain classifier for TinyStories."""
    s = sentence.lower()
    if '"' in s or 'said' in s or 'asked' in s or 'told' in s:
        return 1  # dialogue
    if any(w in s for w in ['happy', 'sad', 'scared', 'excited', 'angry', 'loved', 'felt', 'cried']):
        return 3  # emotion
    if any(w in s for w in ['ran', 'jumped', 'walked', 'flew', 'climbed', 'swam', 'threw', 'caught']):
        return 2  # action
    return 0  # narrative (default)


def load_data(data_path, max_sentences=200000):
    sentences = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if 10 < len(s) < 200:
                sentences.append(s)
            if len(sentences) >= max_sentences:
                break
    random.seed(42)
    random.shuffle(sentences)
    return sentences


def build_vocab(sentences, max_vocab=VOCAB_SIZE):
    word_count = {}
    for s in sentences:
        for w in re.findall(r"[a-zA-Z']+", s.lower()):
            word_count[w] = word_count.get(w, 0) + 1
    sorted_words = sorted(word_count.items(), key=lambda x: -x[1])[:max_vocab - 3]
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2}
    for w, _ in sorted_words:
        vocab[w] = len(vocab)
    return vocab


def tokenize(sentences, vocab, max_len=MAX_LEN):
    bos, eos = vocab['<BOS>'], vocab['<EOS>']
    examples = []
    for s in sentences:
        words = re.findall(r"[a-zA-Z']+", s.lower())
        ids = [vocab[w] for w in words if w in vocab]
        if 4 <= len(ids) <= max_len - 2:
            domain = classify_sentence(s)
            examples.append(([bos] + ids + [eos], domain))
    return examples


def make_batch(examples, indices, max_len=MAX_LEN):
    batch_x, batch_y, domains = [], [], []
    for idx in indices:
        ids, domain = examples[idx]
        x = ids[:-1]
        y = ids[1:]
        pad = max_len - len(x)
        batch_x.append(x + [0] * pad)
        batch_y.append(y + [-100] * pad)
        domains.append(domain)
    return torch.tensor(batch_x), torch.tensor(batch_y), domains


def train(data_path):
    if not Path(data_path).exists():
        print(f"Data not found at {data_path}")
        print("Please provide a valid path to TinyStories-train.txt via --data-path")
        return

    print("Loading data...", flush=True)
    sentences = load_data(str(data_path))
    vocab = build_vocab(sentences)
    examples = tokenize(sentences, vocab)

    val_size = 3000
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    domain_counts = [0] * NUM_EXPERTS
    for _, d in train_examples:
        domain_counts[d] += 1
    print(f"Vocab: {len(vocab)}, Train: {len(train_examples)}, Val: {len(val_examples)}", flush=True)
    for i, name in DOMAINS.items():
        print(f"  Domain {i} ({name}): {domain_counts[i]:,} examples", flush=True)

    model = ExpertLLM(
        vocab_size=len(vocab), hidden_size=HIDDEN, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, head_dim=HEAD_DIM, base_ffn_size=BASE_FFN,
        expert_ffn_size=EXPERT_FFN, num_experts=NUM_EXPERTS
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val = float('inf')
    checkpoint_dir = ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    t0 = time.time()

    for step in range(1, STEPS + 1):
        warmup = STEPS // 20
        if step < warmup:
            lr = LR * step / warmup
        else:
            decay = (step - warmup) / (STEPS - warmup)
            lr = 1e-5 + (LR - 1e-5) * 0.5 * (1 + math.cos(math.pi * decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        optimizer.zero_grad()

        indices = random.sample(range(len(train_examples)), BATCH_SIZE)
        x, y, domains = make_batch(train_examples, indices)

        total_loss = 0.0
        total_tokens = 0

        for domain_id in set(domains):
            mask = [i for i, d in enumerate(domains) if d == domain_id]
            if not mask:
                continue
            bx, by = x[mask], y[mask]
            logits = model(bx, active_experts=[domain_id])
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), by.reshape(-1), ignore_index=-100)
            (loss * len(mask) / BATCH_SIZE).backward()
            total_loss += loss.item() * len(mask)
            total_tokens += len(mask)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_loss = total_loss / max(total_tokens, 1)

        if step % EVAL_INTERVAL == 0:
            model.eval()
            val_losses = {i: [] for i in range(NUM_EXPERTS)}
            val_loss_all = []

            with torch.no_grad():
                for vi in range(0, min(len(val_examples), 2000), BATCH_SIZE):
                    vindices = list(range(vi, min(vi + BATCH_SIZE, len(val_examples))))
                    vx, vy, vdomains = make_batch(val_examples, vindices)
                    for domain_id in set(vdomains):
                        vmask = [i for i, d in enumerate(vdomains) if d == domain_id]
                        if not vmask:
                            continue
                        logits = model(vx[vmask], active_experts=[domain_id])
                        vl = F.cross_entropy(logits.reshape(-1, len(vocab)), vy[vmask].reshape(-1), ignore_index=-100).item()
                        val_losses[domain_id].append(vl)
                        val_loss_all.append(vl)

            val_avg = sum(val_loss_all) / len(val_loss_all) if val_loss_all else 0
            saved = ''
            if val_avg < best_val:
                best_val = val_avg
                torch.save(model.state_dict(), checkpoint_dir / 'expert_model.pt')
                saved = ' *saved*'

            elapsed = (time.time() - t0) / 60
            domain_str = " | ".join(
                f"{DOMAINS[i]}={sum(val_losses[i])/max(len(val_losses[i]),1):.3f}"
                for i in range(NUM_EXPERTS) if val_losses[i]
            )
            print(f"step {step}/{STEPS} | lr {lr:.2e} | train {avg_loss:.4f} | val {val_avg:.4f}{saved} | {domain_str} | {elapsed:.0f}min", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone! Best val: {best_val:.4f} ({elapsed:.1f} min)", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/TinyStories-train.txt',
                        help='Path to TinyStories-train.txt')
    args = parser.parse_args()
    train(args.data_path)
