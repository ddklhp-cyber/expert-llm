"""
Expert Transformer: shared attention + base FFN + domain expert FFNs.
All trained jointly with selective gradient routing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim, max_len=128):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, positions):
        freqs = torch.outer(positions[0].float(), self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2).expand_as(x[..., :x.shape[-1]//2])
        sin = freqs.sin().unsqueeze(0).unsqueeze(2).expand_as(x[..., :x.shape[-1]//2])
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rope = RoPE(head_dim)

    def forward(self, hidden, positions, mask):
        B, T, _ = hidden.shape
        Q = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(hidden).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(hidden).view(B, T, self.num_heads, self.head_dim)
        Q, K = self.rope(Q, positions), self.rope(K, positions)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim) + mask
        output = torch.matmul(F.softmax(scores, dim=-1), V)
        return self.o_proj(output.transpose(1, 2).contiguous().view(B, T, -1))


class FFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertLayer(nn.Module):
    """One transformer layer with shared attention + base FFN + expert FFNs."""

    def __init__(self, hidden_size, num_heads, head_dim, base_ffn_size, expert_ffn_size, num_experts):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, head_dim)
        self.ln2 = RMSNorm(hidden_size)
        self.ffn_base = FFN(hidden_size, base_ffn_size)
        self.experts = nn.ModuleList([FFN(hidden_size, expert_ffn_size) for _ in range(num_experts)])

    def forward(self, hidden, positions, mask, active_experts=None):
        # Shared attention
        hidden = hidden + self.attention(self.ln1(hidden), positions, mask)

        # Base FFN (always active)
        ffn_out = self.ffn_base(self.ln2(hidden))

        # Add active expert FFNs
        if active_experts is not None:
            ln2_out = self.ln2(hidden)
            for expert_id in active_experts:
                ffn_out = ffn_out + self.experts[expert_id](ln2_out)

        hidden = hidden + ffn_out
        return hidden


class ExpertLLM(nn.Module):
    """
    Modular LLM with domain experts.

    Args:
        vocab_size: vocabulary size
        hidden_size: hidden dimension (default 64)
        num_layers: number of transformer layers (default 6)
        num_heads: attention heads (default 8)
        head_dim: dimension per head (default 8)
        base_ffn_size: base FFN intermediate size (default 128)
        expert_ffn_size: expert FFN intermediate size (default 64, smaller than base)
        num_experts: number of domain experts (default 4)
    """

    def __init__(self, vocab_size, hidden_size=64, num_layers=6, num_heads=8,
                 head_dim=8, base_ffn_size=128, expert_ffn_size=64, num_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ExpertLayer(hidden_size, num_heads, head_dim, base_ffn_size, expert_ffn_size, num_experts)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tie embeddings

    def forward(self, input_ids, active_experts=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=input_ids.device), diagonal=1)

        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions, mask, active_experts)

        return self.head(self.norm(hidden))

    def count_params(self, active_experts=None):
        """Count params for a given expert configuration."""
        total = sum(p.numel() for p in self.embed.parameters())
        total += sum(p.numel() for p in self.norm.parameters())
        # head is tied, don't count
        for layer in self.layers:
            total += sum(p.numel() for p in layer.ln1.parameters())
            total += sum(p.numel() for p in layer.attention.parameters())
            total += sum(p.numel() for p in layer.ln2.parameters())
            total += sum(p.numel() for p in layer.ffn_base.parameters())
            if active_experts:
                for eid in active_experts:
                    total += sum(p.numel() for p in layer.experts[eid].parameters())
        return total


if __name__ == '__main__':
    # Quick test
    model = ExpertLLM(vocab_size=5007, num_experts=4)
    total = sum(p.numel() for p in model.parameters())
    base_only = model.count_params(active_experts=None)
    one_expert = model.count_params(active_experts=[0])
    two_experts = model.count_params(active_experts=[0, 1])

    print(f"Total params (all experts): {total:,}")
    print(f"Base only (no experts):     {base_only:,}")
    print(f"Base + 1 expert:            {one_expert:,}")
    print(f"Base + 2 experts:           {two_experts:,}")
    print()

    # Test forward
    x = torch.randint(0, 5007, (2, 16))
    out_base = model(x, active_experts=None)
    out_math = model(x, active_experts=[0])
    out_multi = model(x, active_experts=[0, 2])
    print(f"Output shape: {out_base.shape}")
    print(f"Base vs math differ: {(out_base - out_math).abs().mean():.4f}")
    print(f"Math vs multi differ: {(out_math - out_multi).abs().mean():.4f}")
