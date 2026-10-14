"""Token shard format (llm.c-compatible) and a minimal sequential batch loader.

Shard = 256 x int32 header [magic=20240520, version=1, num_tokens, 0...] + num_tokens x uint16.
Self-check: python data/loader.py --selftest
"""
import glob
import sys

import numpy as np
import torch

MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256


def shard_bytes(tokens):
    """Header + uint16 token bytes for one shard (tokens: uint16 array)."""
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:3] = MAGIC, VERSION, len(tokens)
    return header.tobytes() + np.asarray(tokens, dtype=np.uint16).tobytes()


def read_shard(path):
    """Memmap a shard's tokens, checking the header."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC and header[1] == VERSION, f"bad header in {path}"
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4)
    assert len(tokens) == header[2], f"{path}: header says {header[2]} tokens, file has {len(tokens)}"
    return tokens


class ShardLoader:
    """Yields (x, y) LongTensors of shape (B, T), y = x shifted by one, walking shards in order and cycling.
    Batches never span shards (tail of each shard shorter than B*T+1 is skipped), as in llm.c."""

    def __init__(self, pattern, B, T):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"no shards match {pattern}"
        self.B, self.T = B, T
        self.reset()

    def reset(self):
        self.shard = 0
        self.tokens = read_shard(self.files[0])
        self.pos = 0

    def next_batch(self):
        n = self.B * self.T + 1
        while self.pos + n > len(self.tokens):
            self.shard = (self.shard + 1) % len(self.files)
            self.tokens = read_shard(self.files[self.shard])
            self.pos = 0
            assert len(self.tokens) >= n, f"{self.files[self.shard]} smaller than one batch"
        buf = torch.from_numpy(self.tokens[self.pos:self.pos + n].astype(np.int64))
        self.pos += n - 1
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)


def _selftest():
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        a = np.arange(0, 50, dtype=np.uint16)
        b = np.arange(1000, 1037, dtype=np.uint16)
        for name, toks in [("t_000001.bin", a), ("t_000002.bin", b)]:
            with open(os.path.join(d, name), "wb") as f:
                f.write(shard_bytes(toks))
        header = np.fromfile(os.path.join(d, "t_000001.bin"), dtype=np.int32, count=HEADER_INTS)
        assert list(header[:3]) == [MAGIC, VERSION, 50] and not header[3:].any()
        assert os.path.getsize(os.path.join(d, "t_000001.bin")) == 1024 + 2 * 50
        assert np.array_equal(read_shard(os.path.join(d, "t_000001.bin")), a)

        L = ShardLoader(os.path.join(d, "t_*.bin"), B=2, T=8)  # 17 tokens per batch, stride 16
        x, y = L.next_batch()
        assert x.dtype == torch.long and x.shape == (2, 8)
        assert torch.equal(x.flatten(), torch.arange(0, 16)) and torch.equal(y.flatten(), torch.arange(1, 17))
        L.next_batch(); L.next_batch()          # pos 16, 32 -> next starts at 48, needs 17 > 50: switch shard
        x, y = L.next_batch()
        assert x[0, 0] == 1000 and torch.equal(y.flatten(), x.flatten() + 1), "should move to shard 2"
        L.next_batch()                          # 1016..1032
        x, _ = L.next_batch()                   # shard 2 exhausted -> cycle back to shard 1
        assert x[0, 0] == 0
        L.reset()
        assert L.next_batch()[0][0, 0] == 0
        del L  # Windows can't delete a file that is still memmapped
    print("loader selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
