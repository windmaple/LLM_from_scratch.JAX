"""Orbax checkpointing for the NNX miniGPT (+ optimizer state for resuming).

Layout:
    out/ckpt/model/  Orbax checkpoint of the model params
    out/ckpt/opt/    Orbax checkpoint of the optimizer state (only needed to resume training)
    out/ckpt.json    model config, iteration and training args
"""

import json
import os
import shutil
from dataclasses import asdict

import orbax.checkpoint as ocp
from flax import nnx

from model import MiniGPT, MiniGPTConfig


def _paths(out_dir):
    root = os.path.abspath(os.path.join(out_dir, "ckpt"))
    return os.path.join(root, "model"), os.path.join(root, "opt"), os.path.join(out_dir, "ckpt.json")


def save_checkpoint(out_dir, model, optimizer, cfg, it, args, extra=None):
    """`extra`: optional JSON-serializable dict stored in ckpt.json (e.g. the Grain iterator state)."""
    model_path, opt_path, meta_path = _paths(out_dir)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(model_path, nnx.to_pure_dict(nnx.state(model, nnx.Param)), force=True)
    if optimizer is not None:
        ckptr.save(opt_path, nnx.to_pure_dict(nnx.state(optimizer)), force=True)
    elif os.path.exists(opt_path):
        shutil.rmtree(opt_path)  # don't leave a stale optimizer state next to new params
    ckptr.wait_until_finished()
    with open(meta_path, "w") as f:
        json.dump({"config": asdict(cfg), "iter": it, "args": args, **(extra or {})}, f, indent=1)


def load_meta(out_dir):
    with open(_paths(out_dir)[2]) as f:
        return json.load(f)


def _restore_into(ckptr, path, module, state):
    restored = ckptr.restore(path, nnx.to_pure_dict(state))
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(module, state)


def restore_checkpoint(out_dir, model, optimizer=None):
    """Load params (and optimizer state, if given) in place. Returns the checkpoint metadata."""
    model_path, opt_path, _ = _paths(out_dir)
    ckptr = ocp.StandardCheckpointer()
    _restore_into(ckptr, model_path, model, nnx.state(model, nnx.Param))
    if optimizer is not None:
        _restore_into(ckptr, opt_path, optimizer, nnx.state(optimizer))
    return load_meta(out_dir)


def load_model(out_dir, seed=0):
    """Build a MiniGPT from a checkpoint directory (for sampling / visualization)."""
    meta = load_meta(out_dir)
    model = MiniGPT(MiniGPTConfig(**meta["config"]), rngs=nnx.Rngs(seed))
    restore_checkpoint(out_dir, model)
    return model, meta
