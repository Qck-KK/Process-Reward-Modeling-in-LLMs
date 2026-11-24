"""In-memory store of a step-embedding cache, with batches decoupled from shards.

`precompute_embeddings.py` writes one shard per encoder batch (16 trajectories),
and training used to read one shard from disk per optimiser step. This module
loads every shard once into flat tensors -- all valid steps of all trajectories
in one (n_steps, hidden) tensor plus per-trajectory offsets and lengths -- and
builds padded batches from them:

* `shard_batches(order)` rebuilds the original shards exactly (same rows, same
  padded width, same values), so training on it reproduces the shard loader
  bit for bit while skipping the per-step `torch.load`;
* `random_batches(batch_size, generator)` draws batches of any size over
  trajectories, padded to the longest trajectory in the batch.

The merged store can be saved and memory-mapped (`save` / `load(mmap=True)`),
so a cache larger than RAM is paged in from one file instead of thousands.
"""

import glob
import os

import torch

STORE_VERSION = 1


class CachedSteps:
    def __init__(self, hidden, labels, offsets, lengths, shard_starts, shard_widths):
        self.hidden = hidden                # (total_steps, H), cache dtype (fp16)
        self.labels = labels                # (total_steps,), long, may hold -100
        self.offsets = offsets              # (n_trajectories,) start of each trajectory
        self.lengths = lengths              # (n_trajectories,) valid steps per trajectory
        self.shard_starts = shard_starts    # (n_shards + 1,) trajectory index bounds
        self.shard_widths = shard_widths    # (n_shards,) padded step width of each shard

    # ------------------------------------------------------------------ build
    @classmethod
    def from_shards(cls, shard_paths):
        """Merge shards; refuses shards the flat layout cannot rebuild exactly."""
        hidden, labels, lengths, widths, starts = [], [], [], [], [0]
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu")
            step_hidden = shard["step_hidden"]
            step_mask = shard["step_mask"].bool()
            shard_labels = shard["labels"].long()
            n_rows, width = step_mask.shape
            if shard_labels.shape != step_mask.shape:
                raise ValueError("{0}: labels {1} vs mask {2}".format(
                    path, tuple(shard_labels.shape), tuple(step_mask.shape)))
            row_lengths = step_mask.sum(dim=1)
            prefix = torch.arange(width).unsqueeze(0) < row_lengths.unsqueeze(1)
            padding = ~step_mask
            if (not torch.equal(prefix, step_mask)
                    or step_hidden[padding].abs().sum() != 0
                    or (shard_labels[padding] != -100).any()):
                raise ValueError(
                    "{0}: step_mask is not a left-aligned prefix or padding is not "
                    "zero / -100; use --loader shards for this cache".format(path))
            hidden.append(step_hidden[step_mask])
            labels.append(shard_labels[step_mask])
            lengths.append(row_lengths)
            widths.append(width)
            starts.append(starts[-1] + n_rows)
        lengths = torch.cat(lengths).long()
        offsets = torch.cumsum(lengths, 0) - lengths
        return cls(torch.cat(hidden), torch.cat(labels), offsets, lengths,
                   torch.tensor(starts, dtype=torch.long), torch.tensor(widths, dtype=torch.long))

    @classmethod
    def from_cache_dir(cls, cache_dir, merged_path=None, mmap=False):
        """Load `merged_path` if it exists, else merge the shards (and save if asked)."""
        if merged_path and os.path.exists(merged_path):
            return cls.load(merged_path, mmap=mmap)
        shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
        if not shard_paths:
            raise FileNotFoundError("No cached shards in " + cache_dir)
        store = cls.from_shards(shard_paths)
        if merged_path:
            store.save(merged_path)
            if mmap:
                return cls.load(merged_path, mmap=True)
        return store

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"version": STORE_VERSION, "hidden": self.hidden, "labels": self.labels,
                    "offsets": self.offsets, "lengths": self.lengths,
                    "shard_starts": self.shard_starts, "shard_widths": self.shard_widths}, path)

    @classmethod
    def load(cls, path, mmap=False):
        data = torch.load(path, map_location="cpu", mmap=mmap)
        if data.get("version") != STORE_VERSION:
            raise ValueError("{0}: unsupported merged-cache version".format(path))
        return cls(data["hidden"], data["labels"], data["offsets"], data["lengths"],
                   data["shard_starts"], data["shard_widths"])

    # ------------------------------------------------------------- properties
    @property
    def n_trajectories(self):
        return int(self.lengths.numel())

    @property
    def n_shards(self):
        return int(self.shard_widths.numel())

    @property
    def hidden_size(self):
        return int(self.hidden.shape[1])

    def to(self, device):
        """Move the store (e.g. to the GPU, so batches never cross the bus)."""
        return CachedSteps(*(t.to(device) for t in (
            self.hidden, self.labels, self.offsets, self.lengths,
            self.shard_starts, self.shard_widths)))

    # ---------------------------------------------------------------- batches
    def batch(self, rows, width=None):
        """Padded (step_hidden, step_mask, labels) for trajectory indices `rows`."""
        rows = rows.to(self.lengths.device)
        lengths = self.lengths[rows]
        if width is None:
            width = max(int(lengths.max()) if rows.numel() else 1, 1)
        steps = torch.arange(width, device=lengths.device)
        step_mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
        positions = self.offsets[rows].unsqueeze(1) + steps.unsqueeze(0)
        flat = positions[step_mask]
        step_hidden = self.hidden.new_zeros(rows.numel(), width, self.hidden.shape[1])
        step_hidden[step_mask] = self.hidden[flat]
        labels = torch.full((rows.numel(), width), -100, dtype=torch.long,
                            device=self.labels.device)
        labels[step_mask] = self.labels[flat]
        return step_hidden, step_mask, labels

    def shard_batch(self, index):
        start, end = int(self.shard_starts[index]), int(self.shard_starts[index + 1])
        rows = torch.arange(start, end, device=self.lengths.device)
        return self.batch(rows, width=int(self.shard_widths[index]))

    def shard_batches(self, order=None):
        for index in (range(self.n_shards) if order is None else order):
            yield self.shard_batch(index)

    def random_batches(self, batch_size, generator=None):
        order = torch.randperm(self.n_trajectories, generator=generator)
        for start in range(0, self.n_trajectories, batch_size):
            yield self.batch(order[start:start + batch_size])

    def n_batches(self, batch_size=None):
        if not batch_size:
            return self.n_shards
        return (self.n_trajectories + batch_size - 1) // batch_size
