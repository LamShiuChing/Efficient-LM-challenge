"""Model adapters: each returns (logits_fn, encode) for evals.harness.evaluate."""
import torch
import tiktoken

import evals.harness  # noqa: F401  (sets HF_HOME before transformers is imported)

GPT2_VOCAB = 50257


def _device(device):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _encode():
    return tiktoken.get_encoding("gpt2").encode_ordinary


def hf_gpt2(name="gpt2", device=None):
    from transformers import GPT2LMHeadModel
    device = _device(device)
    model = GPT2LMHeadModel.from_pretrained(name).to(device).eval()
    return (lambda x: model(x.to(device)).logits[..., :GPT2_VOCAB]), _encode()


def local_ckpt(path, device=None):
    from src.model import load_checkpoint
    device = _device(device)
    model = load_checkpoint(path, device=device)
    return (lambda x: model(x.to(device))[0][..., :GPT2_VOCAB]), _encode()
