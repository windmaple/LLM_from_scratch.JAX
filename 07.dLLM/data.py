"""TinyStories data preparation + a compact tokenizer.

We reuse GPT-2's BPE (via `tiktoken`, a pure tokenizer library -- no model
code), but remap it to a compact vocabulary consisting of the most frequent
GPT-2 tokens in TinyStories. TinyStories has a tiny vocabulary, so ~8k tokens
cover >99.9% of the corpus while making the embedding / softmax ~6x cheaper,
which matters a lot when training on CPU.

Special tokens:
    0 = <eos>   (end of story; also used as right-padding)
    1 = <mask>  (the absorbing state of the diffusion process)
    2 = <unk>   (rare GPT-2 tokens that did not make the cut)

Usage:
    python data.py                      # writes data/{train,val}.bin + data/vocab.json
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import tiktoken

EOS_ID, MASK_ID, UNK_ID = 0, 1, 2
N_SPECIAL = 3


class CompactTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            gpt2_ids = json.load(f)["gpt2_ids"]  # compact id (>= N_SPECIAL) -> gpt2 id
        self.enc = tiktoken.get_encoding("gpt2")
        self.to_gpt2 = np.array([-1] * N_SPECIAL + gpt2_ids, dtype=np.int64)
        self.from_gpt2 = np.full(self.enc.n_vocab, UNK_ID, dtype=np.int64)
        self.from_gpt2[gpt2_ids] = np.arange(N_SPECIAL, N_SPECIAL + len(gpt2_ids))
        self.vocab_size = len(self.to_gpt2)

    def encode(self, text):
        return self.from_gpt2[self.enc.encode_ordinary(text)].tolist()

    def encode_batch(self, texts, num_threads=32):
        out = self.enc.encode_ordinary_batch(texts, num_threads=num_threads)
        return [self.from_gpt2[ids] for ids in out]

    def token_str(self, i):
        """Human-readable string of a single compact token id."""
        if i == EOS_ID:
            return "<eos>"
        if i == MASK_ID:
            return "<mask>"
        if i == UNK_ID:
            return "<unk>"
        return self.enc.decode_single_token_bytes(int(self.to_gpt2[i])).decode("utf-8", errors="replace")

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID and stop_at_eos:
                break
            if i >= N_SPECIAL:
                out.append(int(self.to_gpt2[i]))
        return self.enc.decode(out)


def find_parquets(split):
    pat = os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/*/data/{split}-*.parquet")
    files = sorted(glob.glob(pat))
    if not files:  # not cached locally -> let HF download it
        from huggingface_hub import snapshot_download
        root = snapshot_download("roneneldan/TinyStories", repo_type="dataset",
                                 allow_patterns=["data/*.parquet"])
        files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
    return files


def load_texts(split, max_files=None):
    files = find_parquets(split)[:max_files]
    texts = []
    for f in files:
        texts.extend(pq.read_table(f).column("text").to_pylist())
    return [t.strip() for t in texts if t and t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--train_files", type=int, default=1, help="# of the 4 train parquet shards to use")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    splits = {"train": load_texts("train", args.train_files), "val": load_texts("validation")}
    toks = {}
    for split, texts in splits.items():
        print(f"{split}: {len(texts):,} stories, tokenizing...")
        toks[split] = enc.encode_ordinary_batch(texts, num_threads=os.cpu_count())

    # Build the compact vocabulary from training-set frequencies.
    counts = np.bincount(np.concatenate([np.asarray(t) for t in toks["train"]]), minlength=enc.n_vocab)
    keep = np.argsort(-counts)[: args.vocab_size - N_SPECIAL]
    keep = keep[counts[keep] > 0]
    coverage = counts[keep].sum() / counts.sum()
    print(f"compact vocab: {len(keep) + N_SPECIAL} tokens, covers {coverage:.4%} of train tokens")
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"gpt2_ids": keep.tolist()}, f)

    tok = CompactTokenizer(vocab_path)
    for split, seqs in toks.items():
        # Flat stream of stories, each terminated by <eos>; plus start offsets.
        seqs = [np.append(tok.from_gpt2[np.asarray(s, dtype=np.int64)], EOS_ID) for s in seqs]
        lens = np.array([len(s) for s in seqs])
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        np.concatenate(seqs).astype(np.uint16).tofile(os.path.join(args.out_dir, f"{split}.bin"))
        np.save(os.path.join(args.out_dir, f"{split}_offsets.npy"), offsets.astype(np.int64))
        print(f"{split}: {lens.sum():,} tokens | story len mean {lens.mean():.0f}, "
              f"median {np.median(lens):.0f}, p90 {np.percentile(lens, 90):.0f}, "
              f"<=256: {(lens <= 256).mean():.1%}")


if __name__ == "__main__":
    main()
