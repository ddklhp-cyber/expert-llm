"""
Generate text using trained ExpertLLM checkpoints.

Usage:
    # Synthetic model (6 domains)
    python generate.py --checkpoint checkpoints/expert_synthetic.pt --prompt "the bird"

    # TinyStories model (4 domains)
    python generate.py --checkpoint checkpoints/expert_model.pt --num-experts 4 --prompt "the cat"

    # Specify expert(s)
    python generate.py --checkpoint checkpoints/expert_synthetic.pt --experts 1 3 --prompt "water boils"
"""
import sys
import json
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.expert_transformer import ExpertLLM

SYNTH_DOMAINS = {0: "reading", 1: "math", 2: "nature", 3: "science", 4: "history", 5: "code"}
TINY_DOMAINS = {0: "narrative", 1: "dialogue", 2: "action", 3: "emotion"}


def load_vocab(checkpoint_path):
    """Load vocab JSON matching the checkpoint."""
    cp = Path(checkpoint_path)
    # Try matching vocab file
    if 'synthetic' in cp.name:
        vocab_path = cp.parent / 'vocab_synthetic.json'
    else:
        vocab_path = cp.parent / 'vocab_tinystories.json'
    if not vocab_path.exists():
        # Fallback: try any vocab file in same dir
        for f in cp.parent.glob('vocab_*.json'):
            vocab_path = f
            break
    with open(vocab_path) as f:
        return json.load(f)


def generate(model, vocab, rev_vocab, prompt_words, active_experts, max_new=20):
    bos = vocab.get('<BOS>', 1)
    ids = [bos] + [vocab[w] for w in prompt_words if w in vocab]
    for _ in range(max_new):
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = model(x, active_experts=active_experts)
        next_id = logits[0, -1].argmax().item()
        if next_id == vocab.get('<EOS>', 2):
            break
        ids.append(next_id)
    return ' '.join(rev_vocab.get(i, '?') for i in ids[1:])


def main():
    parser = argparse.ArgumentParser(description='Generate text with ExpertLLM')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/expert_synthetic.pt')
    parser.add_argument('--num-experts', type=int, default=None, help='Number of experts (auto-detected)')
    parser.add_argument('--experts', type=int, nargs='*', default=None, help='Expert IDs to activate')
    parser.add_argument('--prompt', type=str, default='the bird', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=20)
    args = parser.parse_args()

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location='cpu')
    vocab_size = state['embed.weight'].shape[0]

    # Auto-detect num_experts from checkpoint
    import re
    expert_keys = [k for k in state.keys() if 'experts' in k]
    num_experts = args.num_experts or (max(int(re.search(r'experts\.(\d+)', k).group(1)) for k in expert_keys) + 1)

    # Build model
    model = ExpertLLM(vocab_size=vocab_size, num_experts=num_experts)
    model.load_state_dict(state)
    model.eval()

    # Load vocab
    vocab = load_vocab(args.checkpoint)
    rev_vocab = {int(v): k for k, v in vocab.items()}

    # Pick domain names
    domains = SYNTH_DOMAINS if num_experts == 6 else TINY_DOMAINS

    print(f"Loaded {args.checkpoint} (vocab={vocab_size}, experts={num_experts}, params={sum(p.numel() for p in model.parameters()):,})")
    print(f"Prompt: \"{args.prompt}\"")
    print()

    prompt_words = args.prompt.lower().split()

    if args.experts is not None:
        names = '+'.join(domains.get(e, str(e)) for e in args.experts)
        text = generate(model, vocab, rev_vocab, prompt_words, args.experts, args.max_tokens)
        print(f"  [{names}]: {text}")
    else:
        for eid in range(num_experts):
            text = generate(model, vocab, rev_vocab, prompt_words, [eid], args.max_tokens)
            print(f"  [{domains.get(eid, str(eid))}]: {text}")
        print()
        text = generate(model, vocab, rev_vocab, prompt_words, None, args.max_tokens)
        print(f"  [base only]: {text}")


if __name__ == '__main__':
    main()
