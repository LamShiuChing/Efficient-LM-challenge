"""FineWeb sample-10BT -> GPT-2 token shards in llm.c's .bin format.

Reproduces llm.c dev/data/fineweb.py (classic, 10B, gpt-2): same docs in the same order, <|endoftext|> (50256)
prepended to every doc, 1e8-token shards, documents split across shard boundaries, shard 0 = val.
Reads the parquet files directly with pyarrow (in sorted file order, which is the order load_dataset uses)
instead of load_dataset, to avoid a second ~40 GB Arrow copy. Raw parquet stays in the HF cache.

Full run:  python data/prepare_fineweb.py
Quick test: python data/prepare_fineweb.py --out_dir data_cache/fineweb_test --max_tokens 20000000 --shard_size 5000000
Re-running continues after the last shard recorded in manifest.json.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data_cache", "hf"))  # must precede huggingface imports

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

from loader import shard_bytes

REPO, CONFIG, SUBDIR = "HuggingFaceFW/fineweb", "sample-10BT", "sample/10BT/"
EOT = 50256

_enc = None


def tokenize(texts):
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    out = []
    for t in texts:
        toks = np.array([EOT] + _enc.encode_ordinary(t), dtype=np.int64)
        assert toks.max() < 2**16
        out.append(toks.astype(np.uint16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data_cache", "fineweb10B"))
    ap.add_argument("--revision", default=None, help="dataset commit sha (default: current main, recorded in manifest)")
    ap.add_argument("--shard_size", type=int, default=10**8)
    ap.add_argument("--max_tokens", type=int, default=None, help="stop after this many tokens (quick test)")
    ap.add_argument("--limit_docs", type=int, default=None, help="stop after this many documents (quick test)")
    ap.add_argument("--nprocs", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")

    api = HfApi()
    if os.path.exists(manifest_path):
        man = json.load(open(manifest_path))
        settings = {"shard_size": args.shard_size, "max_tokens": args.max_tokens, "limit_docs": args.limit_docs}
        assert man["settings"] == settings, f"settings differ from existing manifest: {man['settings']} vs {settings}"
        assert args.revision in (None, man["revision"]), "revision differs from existing manifest"
        for s in man["shards"]:  # cheap integrity check; sha256 is in the manifest for a full check
            p = os.path.join(args.out_dir, s["file"])
            assert os.path.getsize(p) == 1024 + 2 * s["num_tokens"], f"{p} missing or truncated; delete it + manifest"
        if man["complete"]:
            print(f"{manifest_path} says complete ({len(man['shards'])} shards); nothing to do")
            return
        print(f"resuming after {len(man['shards'])} shards")
    else:
        info = api.dataset_info(REPO, revision=args.revision or "main", files_metadata=True)
        files = sorted((s.rfilename, s.lfs.sha256, s.size) for s in info.siblings if s.rfilename.startswith(SUBDIR))
        man = {
            "dataset": REPO, "config": CONFIG, "revision": info.sha,
            "source_files": [{"path": f, "sha256": h, "size": n} for f, h, n in files],
            "tokenizer": "tiktoken gpt2 (encode_ordinary), <|endoftext|>=50256 prepended to each doc",
            "format": "256 x int32 header [20240520, 1, num_tokens] + uint16 tokens (llm.c)",
            "settings": {"shard_size": args.shard_size, "max_tokens": args.max_tokens, "limit_docs": args.limit_docs},
            "shards": [], "complete": False,
        }
    srcs = [f["path"] for f in man["source_files"]]
    # resume point: (file index, row in file, tokens of that doc already written)
    fi, row0, skip = man["shards"][-1]["end"] if man["shards"] else (0, 0, 0)
    total_tokens = sum(s["num_tokens"] for s in man["shards"])
    total_docs = man["shards"][-1]["docs_started"] if man["shards"] else 0

    buf = np.empty(args.shard_size, dtype=np.uint16)
    n = 0  # tokens in buf

    def save():  # atomic, so a kill never leaves a half-written manifest
        with open(manifest_path + ".tmp", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(manifest_path + ".tmp", manifest_path)

    def flush(end):
        idx = len(man["shards"])
        name = f"fineweb_{'val' if idx == 0 else 'train'}_{idx:06d}.bin"
        data = shard_bytes(buf[:n])
        tmp = os.path.join(args.out_dir, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(args.out_dir, name))
        man["shards"].append({"file": name, "num_tokens": int(n), "sha256": hashlib.sha256(data).hexdigest(),
                              "end": list(end), "docs_started": total_docs})
        save()
        print(f"wrote {name}: {n:,} tokens | total {total_tokens:,} tokens, {total_docs:,} docs | "
              f"{(total_tokens - tokens0) / (time.time() - t0) / 1e6:.2f} M tok/s (incl. download)", flush=True)

    download = lambda f: hf_hub_download(REPO, f, repo_type="dataset", revision=man["revision"])
    t0, tokens0 = time.time(), total_tokens
    done = False
    with mp.Pool(args.nprocs) as pool, ThreadPoolExecutor(1) as prefetch:
        nxt = prefetch.submit(download, srcs[fi])
        full_run = args.max_tokens is None and args.limit_docs is None
        for fi in range(fi, len(srcs)):
            t = time.time()
            path = nxt.result()
            print(f"[{fi + 1}/{len(srcs)}] {srcs[fi]} (waited {time.time() - t:.0f}s for download)", flush=True)
            if full_run and fi + 1 < len(srcs):
                nxt = prefetch.submit(download, srcs[fi + 1])  # overlap next download with tokenizing this one
            pf = pq.ParquetFile(path)
            row = 0
            # ponytail: Pool.map per 64k-doc batch (bounded memory); workers idle briefly between batches
            for rb in pf.iter_batches(batch_size=65536, columns=["text"]):
                texts = rb.column(0).to_pylist()
                if row + len(texts) <= row0:
                    row += len(texts)
                    continue
                texts, row = texts[max(0, row0 - row):], max(row, row0)
                if args.limit_docs is not None:
                    texts = texts[:args.limit_docs - total_docs]
                chunks = [texts[i:i + 256] for i in range(0, len(texts), 256)]
                for toks in (t for c in pool.map(tokenize, chunks) for t in c):
                    base, skip = skip, 0  # tokens of this doc written before a resume
                    total_docs += base == 0
                    toks = toks[base:]
                    if args.max_tokens is not None:
                        toks = toks[:args.max_tokens - total_tokens]
                    off = 0  # tokens of toks already placed
                    while n + len(toks) - off >= args.shard_size:  # llm.c: a doc that exactly fills a shard closes it
                        take = args.shard_size - n
                        buf[n:] = toks[off:off + take]
                        n, off, total_tokens = args.shard_size, off + take, total_tokens + take
                        flush((fi, row, base + off))
                        n = 0
                    buf[n:n + len(toks) - off] = toks[off:]
                    total_tokens += len(toks) - off
                    n += len(toks) - off
                    row += 1
                    if (args.max_tokens is not None and total_tokens >= args.max_tokens) or \
                       (args.limit_docs is not None and total_docs >= args.limit_docs):
                        done = True
                        break
                if done:
                    break
            row0 = 0
            if done:
                break
    if n:
        flush((fi, row, 0))
    man["complete"] = True
    save()
    print(f"done: {len(man['shards'])} shards, {total_tokens:,} tokens, {total_docs:,} docs, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
