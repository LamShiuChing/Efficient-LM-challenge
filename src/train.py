"""Single-GPU GPT trainer.

    python src/train.py configs/gpt2_124m_fineweb10B.json [--key=value ...] [--flag]

Config = DEFAULTS <- json file <- CLI overrides. Every key is flat; model shape keys are the
GPTConfig fields. `--flag` alone sets a key to true; values are parsed as JSON, else string.
"""
import dataclasses
import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig, load_checkpoint  # noqa: E402

DEFAULTS = dict(
    run_name="debug",
    seed=1337,
    # data
    train_pattern="data_cache/fineweb10B/fineweb_train_*.bin",
    val_pattern="data_cache/fineweb10B/fineweb_val_*.bin",
    synthetic=False,          # random tokens instead of shards (tests / benchmarks)
    # model (GPTConfig fields)
    **dataclasses.asdict(GPTConfig()),
    # optimization
    total_batch_size=524288,  # tokens per optimizer step; grad accumulation fills it
    micro_batch_size=16,
    max_steps=18865,
    lr=6e-4,
    min_lr_frac=0.0,          # final lr = lr * min_lr_frac
    warmup_steps=700,
    decay="cosine",           # cosine | linear
    weight_decay=0.1,         # 2D params only
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    grad_clip=1.0,
    # eval / io
    val_every=250,
    val_tokens=10_485_760,
    ckpt_every=5000,
    log_every=1,
    compile=True,
    device="cuda",            # "cpu" for GPU-free tests
    peak_flops=0.0,           # GPU dense bf16 peak for MFU; 0 = don't report MFU (set per GPU from its spec sheet)
    exploratory=False,        # allow dirty git tree (not for preregistered runs)
    benchmark_steps=0,        # >0: time this many steps, print throughput, write nothing
)


def parse_config(argv):
    cfg = dict(DEFAULTS)
    args = list(argv)
    if args and not args[0].startswith("--"):
        cfg.update(json.loads(Path(args.pop(0)).read_text()))
    for a in args:
        assert a.startswith("--"), f"bad arg {a}"
        k, _, v = a[2:].partition("=")
        if not _:
            val = True
        else:
            try:
                val = json.loads(v)
            except json.JSONDecodeError:
                val = v
        cfg[k] = val
    unknown = {k for k in cfg if k not in DEFAULTS and not k.startswith("_")}  # "_x" keys = comments
    assert not unknown, f"unknown config keys: {unknown}"
    return cfg


class SyntheticLoader:
    """Random tokens, same interface as data.loader.ShardLoader."""

    def __init__(self, B, T, vocab=50257, seed=0):
        self.B, self.T, self.vocab, self.seed = B, T, vocab, seed
        self.reset()

    def reset(self):
        self.g = torch.Generator().manual_seed(self.seed)

    def next_batch(self):
        buf = torch.randint(0, self.vocab, (self.B, self.T + 1), generator=self.g)
        return buf[:, :-1], buf[:, 1:]


def make_loader(cfg, pattern, B, T, seed):
    if cfg["synthetic"]:
        return SyntheticLoader(B, T, seed=seed)
    from data.loader import ShardLoader  # lazy: owned by the data agent
    return ShardLoader(str(ROOT / pattern), B, T)


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, dirty


def data_manifest(cfg):
    if cfg["synthetic"]:
        return {"synthetic": True, "seed": cfg["seed"]}
    # ponytail: names+sizes, not content hashes (20 GB); hash contents if shards get rewritten in place
    shards = {p: sorted((Path(f).name, os.path.getsize(f)) for f in glob.glob(str(ROOT / cfg[p])))
              for p in ("train_pattern", "val_pattern")}
    sha = hashlib.sha256(json.dumps(shards).encode()).hexdigest()
    man = ROOT / Path(cfg["train_pattern"]).parent / "manifest.json"
    man_sha = hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None
    return {"sha256": sha, "manifest_sha256": man_sha, "n_train_shards": len(shards["train_pattern"]),
            "n_val_shards": len(shards["val_pattern"]), "shards": shards}


def get_lr(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    frac = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    coeff = 0.5 * (1 + math.cos(math.pi * frac)) if cfg["decay"] == "cosine" else 1 - frac
    min_lr = cfg["lr"] * cfg["min_lr_frac"]
    return min_lr + coeff * (cfg["lr"] - min_lr)


def main(argv):
    cfg = parse_config(argv)
    bench = cfg["benchmark_steps"] > 0
    commit, dirty = git_state()
    if dirty and not (cfg["exploratory"] or bench):
        sys.exit("git tree is dirty (or not a repo): commit first, or pass --exploratory (protocol section 8)")

    torch.manual_seed(cfg["seed"])  # seeds CUDA too
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = cfg["device"]
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    B, T = cfg["micro_batch_size"], cfg["block_size"]
    assert cfg["total_batch_size"] % (B * T) == 0, "total_batch_size must be divisible by B*T"
    grad_accum = cfg["total_batch_size"] // (B * T)
    max_steps = cfg["benchmark_steps"] if bench else cfg["max_steps"]

    mcfg = GPTConfig(**{f.name: cfg[f.name] for f in dataclasses.fields(GPTConfig)})
    raw_model = GPT(mcfg).to(device)
    model = torch.compile(raw_model) if cfg["compile"] else raw_model
    n_total, n_nonemb = raw_model.num_params(False), raw_model.num_params(True)
    # MFU flops/token: 6*(matmul params incl. lm_head, excl. wpe) + attention (PaLM appendix B)
    L, H, Q = mcfg.n_layer, mcfg.n_head, mcfg.n_embd // mcfg.n_head
    flops_per_token = 6 * (n_total - raw_model.transformer.wpe.weight.numel()) + 12 * L * H * Q * T
    planned_tokens = max_steps * cfg["total_batch_size"]

    def mfu(tok_s):
        return round(flops_per_token * tok_s / cfg["peak_flops"], 4) if cfg["peak_flops"] else None

    decay = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"], fused=True)

    train_loader = make_loader(cfg, cfg["train_pattern"], B, T, seed=cfg["seed"])
    val_loader = None if bench else make_loader(cfg, cfg["val_pattern"], B, T, seed=cfg["seed"] + 1)
    val_steps = max(1, cfg["val_tokens"] // (B * T))

    run_dir = ROOT / "runs" / cfg["run_name"]
    if not bench:
        if (run_dir / "log.jsonl").exists() and not cfg["exploratory"]:
            sys.exit(f"{run_dir} already has a log; pick a new run_name")
        run_dir.mkdir(parents=True, exist_ok=True)
        info = dict(
            config=cfg, git_commit=commit, git_dirty=dirty, argv=sys.argv,
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if cuda else "cpu", n_gpus=int(cuda), platform=platform.platform(),
            params_total=n_total, params_non_embedding=n_nonemb, grad_accum=grad_accum,
            planned_tokens=planned_tokens, planned_flops_6ND=6 * n_nonemb * planned_tokens,
            flops_per_token_mfu=flops_per_token, data_manifest=data_manifest(cfg),
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        log = open(run_dir / "log.jsonl", "w")
    print(f"params total {n_total/1e6:.2f}M, non-embedding {n_nonemb/1e6:.2f}M | grad_accum {grad_accum} | "
          f"planned {planned_tokens/1e9:.3f}B tokens")

    def write(rec):
        print(json.dumps(rec))
        if not bench:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    def save(step, tokens_seen):
        path = run_dir / f"ckpt_{step:06d}.pt"
        torch.save({"model": raw_model.state_dict(), "config": dataclasses.asdict(mcfg), "step": step,
                    "tokens_seen": tokens_seen, "train_config": cfg}, path)
        return path

    @torch.no_grad()
    def evaluate():
        model.eval()
        val_loader.reset()
        total = 0.0
        for _ in range(val_steps):
            x, y = val_loader.next_batch()
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                total += model(x.to(device), y.to(device))[1].item()
        model.train()
        return total / val_steps

    t_start = time.time()
    tokens_seen, timings = 0, []
    for step in range(max_steps):
        last = step == max_steps - 1
        if val_loader and (step % cfg["val_every"] == 0 or last):
            write({"step": step, "tokens_seen": tokens_seen, "val_loss": evaluate()})
        if not bench and step > 0 and step % cfg["ckpt_every"] == 0:
            save(step, tokens_seen)

        sync()
        t0 = time.time()
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            if cuda:
                x, y = x.pin_memory(), y.pin_memory()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum
            loss_acc += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        dt = time.time() - t0
        tokens_seen += cfg["total_batch_size"]
        tok_s = cfg["total_batch_size"] / dt
        timings.append(dt)
        if step % cfg["log_every"] == 0 or last:
            write({"step": step, "loss": loss_acc.item(), "lr": lr, "grad_norm": norm.item(),
                   "tokens_seen": tokens_seen, "dt_ms": round(dt * 1000, 1), "tok_per_s": round(tok_s),
                   "mfu": mfu(tok_s)})

    if bench:
        steady = timings[min(3, len(timings) - 1):]  # drop compile/warmup steps
        tok_s = cfg["total_batch_size"] * len(steady) / sum(steady)
        print("BENCHMARK " + json.dumps({
            "compile": cfg["compile"], "micro_batch_size": B, "seq_len": T, "grad_accum": grad_accum,
            "tok_per_s": round(tok_s), "mfu": mfu(tok_s),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "hours_per_10B": round(10e9 / tok_s / 3600, 1)}))
        return

    path = save(max_steps, tokens_seen)
    # round-trip check: the checkpoint reproduces the trained model's logits
    x, _ = val_loader.next_batch()
    x = x[:2].to(device)
    with torch.no_grad():
        ref = raw_model.eval()(x)[0]
        got = load_checkpoint(path, device)(x)[0]
    max_diff = (ref - got).abs().max().item()
    assert max_diff < 1e-5, f"checkpoint round-trip mismatch {max_diff}"
    write({"step": max_steps, "tokens_seen": tokens_seen, "wallclock_hours": (time.time() - t_start) / 3600,
           "checkpoint": str(path), "ckpt_roundtrip_max_abs_diff": max_diff})


if __name__ == "__main__":
    main(sys.argv[1:])
