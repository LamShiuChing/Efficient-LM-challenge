"""GPT-2 (pre-LN, GELU, tied embeddings) with learned positional embeddings or RoPE. nanoGPT-style."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # 50257 real gpt2 tokens, padded to a multiple of 64 for speed
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    pos: str = "learned"       # "learned" (GPT-2 table, capped at block_size) | "rope" (rotary, Su et al. 2021)
    rope_base: float = 10000.0


def rope_cache(T, head_dim, base):
    """cos, sin of shape (T, head_dim/2): position t rotates channel pair i by t * base^(-2i/head_dim)."""
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (B, H, T, D): rotate the pairs (x[..., i], x[..., i + D/2]) by the position angle (rotate-half form)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[:x.size(-2)].to(x.dtype), sin[:x.size(-2)].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True

    def forward(self, x, rope=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln_1(x), rope)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        assert cfg.pos in ("learned", "rope"), cfg.pos
        mods = dict(wte=nn.Embedding(cfg.vocab_size, cfg.n_embd))
        if cfg.pos == "learned":  # created in GPT-2's order, so a given seed still gives the same init as before RoPE
            mods["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        mods.update(h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]), ln_f=nn.LayerNorm(cfg.n_embd))
        self.transformer = nn.ModuleDict(mods)
        if cfg.pos == "rope":  # not in the state_dict: recomputed from the config
            cos, sin = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tied
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # GPT-2 / llm.c init: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer)
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layer) if getattr(m, "RESIDUAL_SCALE", False) else 0.02
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        """Non-embedding excludes token and position embeddings (Kaplan et al.); the tied
        lm_head shares the token embedding, so it is excluded too."""
        n = sum(p.numel() for p in self.parameters())  # tied weight counted once
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
            if "wpe" in self.transformer:
                n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        x = self.transformer.wte(idx)
        rope = None
        if self.config.pos == "learned":
            x = x + self.transformer.wpe(torch.arange(T, device=idx.device))
        else:
            rope = (self.rope_cos, self.rope_sin)
        for block in self.transformer.h:
            x = block(x, rope)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss


def load_checkpoint(path, device="cpu") -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval()
