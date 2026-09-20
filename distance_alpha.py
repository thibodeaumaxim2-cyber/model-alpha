"""Experimental causal Alpha with differentiable hierarchical vector memory."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DistanceMemory(nn.Module):
    def __init__(self, dim=256, clusters=64, slots=64, target=8):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(clusters, dim) * .02)
        self.offsets = nn.Parameter(torch.randn(clusters, slots, dim) * .02)
        self.values = nn.Parameter(torch.randn(clusters, slots, dim) * .02)
        self.query = nn.Linear(dim, dim, bias=False)
        self.radius = nn.Linear(dim, 1)
        nn.init.zeros_(self.radius.weight)
        nn.init.constant_(self.radius.bias, -2.)
        self.gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)
        self.target = target

    def forward(self, hidden, chunk_size=128):
        # Chunk queries to bound the [tokens, clusters, slots] score allocation.
        flat = hidden.reshape(-1, hidden.shape[-1])
        centers = F.normalize(self.centers.float(), dim=-1)
        keys = F.normalize((self.centers[:, None] + self.offsets).float(), dim=-1)
        outputs, effective = [], []
        usage = torch.zeros_like(self.values[..., 0], dtype=torch.float32)
        penalty = hidden.new_zeros((), dtype=torch.float32)
        for h in flat.split(chunk_size):
            q = F.normalize(self.query(h).float(), dim=-1)
            radius = .03 + .97 * torch.sigmoid(self.radius(h).float())
            cluster_dist = (2 - 2 * q @ centers.T).clamp_min(0)
            cluster_weights = (-cluster_dist / (2 * radius.square())).softmax(-1)
            distances = (2 - 2 * torch.einsum('bd,csd->bcs', q, keys)).clamp_min(0)
            local = (-distances / (2 * radius.square().unsqueeze(-1))).softmax(-1)
            weights = cluster_weights.unsqueeze(-1) * local
            count = 1 / weights.square().sum((1, 2)).clamp_min(1e-12)
            penalty = penalty + (count.log() - math.log(self.target)).square().sum()
            usage = usage + weights.sum(0)
            effective.append(count.detach())
            content = torch.einsum('bcs,csd->bd', weights, self.values.float())
            outputs.append(h + torch.sigmoid(self.gate(h)) * content.to(h.dtype))
        usage = usage / len(flat)
        # Gentle global load balancing; it does not impose eight hard winners.
        balance = usage.numel() * usage.square().sum() - 1
        auxiliary = penalty / len(flat) + .01 * balance
        return torch.cat(outputs).reshape_as(hidden), auxiliary, torch.cat(effective).mean()


class DistanceAlpha(nn.Module):
    def __init__(self, vocab_size=32000, block_size=512, dim=256, heads=8,
                 layers=4, clusters=64, slots=64):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(block_size, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4*dim, dropout=0,
                                          batch_first=True, norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.memory = DistanceMemory(dim, clusters, slots)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight

    def forward(self, tokens):
        n = tokens.shape[1]
        h = self.token(tokens) + self.position(torch.arange(n, device=tokens.device))
        mask = torch.ones(n, n, device=tokens.device, dtype=torch.bool).triu(1)
        h = self.blocks(h, mask=mask)
        h, auxiliary, effective = self.memory(h)
        return self.output(self.norm(h)), auxiliary, effective
