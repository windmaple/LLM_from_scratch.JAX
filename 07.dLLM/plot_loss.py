"""Plot training / validation curves from out/log.jsonl.

    python plot_loss.py --log out/log.jsonl --out assets/loss.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="out/log.jsonl")
    ap.add_argument("--out", default="assets/loss.png")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.log)]
    tr = [r for r in rows if "train_elbo" in r]
    va = [r for r in rows if "val_elbo" in r]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["iter"] for r in tr], [r["train_elbo"] for r in tr], alpha=0.4, lw=1, label="train ELBO (nats/token)")
    ax.plot([r["iter"] for r in va], [r["val_elbo"] for r in va], "o-", label="val ELBO (nats/token)")
    ax.plot([r["iter"] for r in va], [r["val_ce"] for r in va], "s--", label="val masked-token CE")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_ylim(top=min(6, ax.get_ylim()[1]))
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("miniGPT (Flax NNX) masked diffusion on TinyStories")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
