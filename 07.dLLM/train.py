"""Train a miniGPT (Flax NNX) masked-diffusion LM on TinyStories.

    python train.py                       # default config
    python train.py --max_iters 200 --eval_interval 100   # quick smoke test
"""

import argparse
import itertools
import json
import math
import os
import time

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from checkpoint import restore_checkpoint, save_checkpoint
from data import EOS_ID, CompactTokenizer
from diffusion import diffusion_loss, generate
from model import MiniGPT, MiniGPTConfig, num_params


# ---- data: Grain pipeline ---------------------------------------------------------------

class StorySource(pygrain.RandomAccessDataSource):
    """Grain random-access data source: record i = the token ids of story i (variable length)."""

    def __init__(self, data_dir, split):
        self.bin_path = os.path.join(data_dir, f"{split}.bin")
        self.offsets = np.load(os.path.join(data_dir, f"{split}_offsets.npy"))
        n_tokens = os.path.getsize(self.bin_path) // np.dtype(np.uint16).itemsize
        self.lens = np.diff(np.append(self.offsets, n_tokens))
        self._tokens = None  # memmap opened lazily, so the source stays picklable for Grain workers

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        start = self.offsets[i]
        return np.array(self._tokens[start: start + self.lens[i]], dtype=np.int32)

    def __getstate__(self):
        return {**self.__dict__, "_tokens": None}

    def __repr__(self):  # Grain checks repr(data_source) when restoring an iterator state
        return f"StorySource({self.bin_path!r}, n_stories={len(self)})"


class PadOrTruncate(pygrain.MapTransform):
    """Each sample = one story from its beginning, truncated / right-padded with <eos> to maxlen."""

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def map(self, tokens):
        out = np.full((self.maxlen,), EOS_ID, dtype=np.int32)
        n = min(len(tokens), self.maxlen)
        out[:n] = tokens[:n]
        return out


def make_loader(source, batch_size, maxlen, seed, worker_count=0):
    """Shuffled, endlessly repeating (num_epochs=None) stream of (batch_size, maxlen) int32 batches."""
    sampler = pygrain.IndexSampler(num_records=len(source), shard_options=pygrain.NoSharding(),
                                   shuffle=True, num_epochs=None, seed=seed)
    return pygrain.DataLoader(data_source=source, sampler=sampler,
                              operations=[PadOrTruncate(maxlen), pygrain.Batch(batch_size, drop_remainder=True)],
                              worker_count=worker_count)


# ---- optimizer: Optax ---------------------------------------------------------------------

def make_schedule(args):
    """Linear warmup lr/W -> lr over W steps, then cosine decay lr -> min_lr."""
    warmup = optax.linear_schedule(init_value=args.lr / args.warmup_iters, end_value=args.lr,
                                   transition_steps=max(1, args.warmup_iters - 1))
    cosine = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=max(1, args.max_iters - args.warmup_iters),
                                         alpha=args.min_lr / args.lr)
    return optax.join_schedules([warmup, cosine], boundaries=[args.warmup_iters])


def make_optimizer(model, args, schedule):
    # AdamW with weight decay only on >=2D params (matrices / embeddings), plus global-norm clipping.
    decay_mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=args.weight_decay, mask=decay_mask),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, x0, key):
    def loss_fn(model):
        return diffusion_loss(model, x0, key, training=True)

    (elbo, ce), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    gnorm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return elbo, ce, gnorm


@nnx.jit
def eval_step(model, x0, key):
    return diffusion_loss(model, x0, key, training=False)


def evaluate(model, val_batches):
    key = jax.random.PRNGKey(1234)  # fixed batches + fixed masking noise -> comparable numbers
    elbos, ces = [], []
    for i, x0 in enumerate(val_batches):
        elbo, ce = eval_step(model, x0, jax.random.fold_in(key, i))
        elbos.append(float(elbo))
        ces.append(float(ce))
    return float(np.mean(elbos)), float(np.mean(ces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    # model (the miniGPT from 01.miniGPT)
    ap.add_argument("--num_transformer_blocks", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--feed_forward_dim", type=int, default=256)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimization
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_iters", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # data loading
    ap.add_argument("--num_workers", type=int, default=2, help="Grain worker processes (0 = load in main process)")
    # logging
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)

    tok = CompactTokenizer(os.path.join(args.data_dir, "vocab.json"))
    train_loader = make_loader(StorySource(args.data_dir, "train"), args.batch_size, args.maxlen,
                               seed=args.seed, worker_count=args.num_workers)
    # A fixed set of validation batches, drawn once with a fixed seed.
    val_loader = make_loader(StorySource(args.data_dir, "val"), args.batch_size, args.maxlen, seed=1234)
    val_batches = [jnp.asarray(b) for b in itertools.islice(val_loader, args.eval_batches)]

    cfg = MiniGPTConfig(vocab_size=tok.vocab_size, maxlen=args.maxlen, embed_dim=args.embed_dim,
                        num_heads=args.num_heads, feed_forward_dim=args.feed_forward_dim,
                        num_transformer_blocks=args.num_transformer_blocks, dropout_rate=args.dropout)
    model = MiniGPT(cfg, rngs=nnx.Rngs(args.seed))
    schedule = make_schedule(args)
    optimizer = make_optimizer(model, args, schedule)
    print(f"devices={jax.devices()} | params: {num_params(model) / 1e6:.2f}M")

    train_iter = iter(train_loader)
    start_it = 0
    if args.resume and os.path.exists(os.path.join(args.out_dir, "ckpt.json")):
        meta = restore_checkpoint(args.out_dir, model, optimizer)
        train_iter.set_state(meta["data_state"].encode())  # continue the exact same data stream
        start_it = meta["iter"] + 1
        print(f"resumed from iter {meta['iter']}")

    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "a" if args.resume else "w")

    def save(it):
        save_checkpoint(args.out_dir, model, optimizer, cfg, it, vars(args),
                        extra={"data_state": train_iter.get_state().decode()})

    t0 = time.time()
    for it in range(start_it, args.max_iters):
        x0 = jnp.asarray(next(train_iter))
        elbo, ce, gnorm = train_step(model, optimizer, x0, jax.random.fold_in(key, it))

        if it % args.log_interval == 0:
            elbo, ce, gnorm = float(elbo), float(ce), float(gnorm)  # blocks until the step is done
            lr = float(schedule(it))
            dt = (time.time() - t0) / (args.log_interval if it > start_it else 1)
            t0 = time.time()
            print(f"iter {it:5d} | elbo {elbo:.3f} | masked-ce {ce:.3f} | "
                  f"gnorm {gnorm:.2f} | lr {lr:.2e} | {dt * 1000:.0f} ms/it", flush=True)
            log_f.write(json.dumps({"iter": it, "train_elbo": elbo, "train_ce": ce, "lr": lr}) + "\n")
            log_f.flush()

        if (it > 0 and it % args.eval_interval == 0) or it == args.max_iters - 1:
            v_elbo, v_ce = evaluate(model, val_batches)
            print(f"==> iter {it} | val elbo {v_elbo:.3f} (ppl bound {math.exp(v_elbo):.1f}) | val masked-ce {v_ce:.3f}")
            log_f.write(json.dumps({"iter": it, "val_elbo": v_elbo, "val_ce": v_ce}) + "\n")
            log_f.flush()
            save(it)
            ids, _ = generate(model, tok.encode("Once upon a time"), length=128, steps=64,
                              temperature=0.8, record=False, key=jax.random.PRNGKey(it))
            print("sample:", repr(tok.decode(ids)[:400]), flush=True)
            t0 = time.time()

    save(args.max_iters - 1)
    print("done")


if __name__ == "__main__":
    main()
