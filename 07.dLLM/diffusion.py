"""Masked (absorbing-state) discrete diffusion, a la MDLM / LLaDA, in JAX.

Forward process: at noise level t in [0, 1], every token is independently
replaced by <mask> with probability t. At t=1 the sequence is all masks.

Training objective (continuous-time ELBO of masked diffusion):
    L = E_{t, x_t} [ (1/t) * sum_{i masked} -log p_theta(x_0^i | x_t) ] / L
i.e. a re-weighted masked-LM loss with a random masking ratio.

Reverse process (generation): start from all masks and, over `steps` steps,
progressively commit the model's predictions for some of the masked
positions, keeping committed tokens fixed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from data import MASK_ID, UNK_ID


def diffusion_loss(model, x0, key, eps=1e-3, training=False):
    """x0: (B, L) clean tokens, key: PRNG key. Returns (elbo_loss, masked_token_ce).

    Gathering only the masked positions would give a data-dependent shape, which `jit`
    cannot compile, so logits are computed for every position and the cross-entropy
    is weighted by the 0/1 mask instead.
    """
    B, L = x0.shape
    k_t, k_mask = jax.random.split(key)
    # Stratified sampling of t over the batch -> lower-variance loss estimate.
    u = (jax.random.uniform(k_t, ()) + jnp.arange(B) / B) % 1.0
    t = eps + (1 - eps) * u  # (B,)
    is_masked = jax.random.uniform(k_mask, (B, L)) < t[:, None]
    xt = jnp.where(is_masked, MASK_ID, x0)

    logits = model(xt, training=training)  # (B, L, V)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), x0)  # (B, L)
    m = is_masked.astype(jnp.float32)
    elbo = jnp.sum(ce * m / t[:, None]) / (B * L)
    masked_ce = jnp.sum(ce * m) / jnp.maximum(m.sum(), 1.0)
    return elbo, masked_ce


def _num_to_unmask(n_masked, steps):
    """Linear schedule: how many tokens to reveal at each of `steps` steps."""
    base = n_masked // steps
    counts = np.full((steps,), base, dtype=np.int64)
    counts[: n_masked % steps] += 1
    return counts


@nnx.jit(static_argnames=("strategy", "top_k", "greedy"))
def _denoise_step(model, x, in_block, k, key, temperature, *, strategy, top_k, greedy):
    """One reverse step: predict every position, then commit `k` of the eligible masked ones."""
    logits = model(x, training=False).astype(jnp.float32)  # (1, L, V)
    logits = logits.at[..., MASK_ID].set(-jnp.inf)  # never predict the special tokens
    logits = logits.at[..., UNK_ID].set(-jnp.inf)
    logits = logits / jnp.maximum(temperature, 1e-5)
    if top_k is not None:
        kth = jax.lax.top_k(logits, top_k)[0][..., -1:]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    probs = jax.nn.softmax(logits, axis=-1)
    k_sample, k_score = jax.random.split(key)
    if greedy:
        x0 = jnp.argmax(logits, axis=-1)
    else:
        x0 = jax.random.categorical(k_sample, logits, axis=-1)  # (1, L)
    conf = jnp.take_along_axis(probs, x0[..., None], axis=-1)[..., 0]  # p(chosen token)

    if strategy == "confidence":
        score = conf
    elif strategy == "random":
        score = jax.random.uniform(k_score, conf.shape)
    else:
        raise ValueError(strategy)
    eligible = (x == MASK_ID) & in_block
    score = jnp.where(eligible, score, -jnp.inf)
    # top-k with a traced k: rank every position by score and keep ranks < k.
    rank = jnp.argsort(jnp.argsort(-score[0]))
    reveal = (rank < k)[None, :] & eligible
    x = jnp.where(reveal, x0, x)
    return x, reveal


def generate(model, prompt_ids=None, length=256, steps=128, temperature=1.0, top_k=None,
             strategy="confidence", block_len=None, record=True, key=None):
    """Sample one sequence by iterative unmasking.

    strategy:
      "confidence": at each step reveal the masked positions whose sampled token
                    has the highest probability (LLaDA's low-confidence remasking).
      "random":     reveal a uniformly random subset of masked positions
                    (the vanilla MDLM ancestral sampler).
    block_len: if set, use semi-autoregressive decoding (LLaDA): the sequence is split
      into blocks of `block_len` tokens that are denoised left to right, in parallel
      within each block (the model still attends to the whole sequence). This stops
      confidence-based sampling from committing far-away, easy tokens (e.g. trailing
      <eos> padding) too early.
    Returns (final_tokens, history) where history is a list of (tokens, newly_revealed_mask)
    numpy arrays, one per step (step 0 = initial state).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    x = np.full((1, length), MASK_ID, dtype=np.int32)
    if prompt_ids:
        p = np.asarray(prompt_ids[:length], dtype=np.int32)
        x[0, : len(p)] = p
    history = [(x[0].copy(), np.zeros(length, dtype=bool))] if record else []

    # Partition the masked positions into blocks and give each block a share of the steps.
    masked_pos = np.nonzero(x[0] == MASK_ID)[0]
    if block_len is None:
        blocks = [masked_pos]
    else:
        blocks = [masked_pos[i: i + block_len] for i in range(0, len(masked_pos), block_len)]
    blocks = [b for b in blocks if len(b)]
    steps_per_block = max(1, steps // max(1, len(blocks)))

    x = jnp.asarray(x)
    temperature = jnp.float32(temperature)
    for blk in blocks:
        in_block = np.zeros((1, length), dtype=bool)
        in_block[0, blk] = True
        in_block = jnp.asarray(in_block)
        n_steps = min(steps_per_block, len(blk))
        for k in _num_to_unmask(len(blk), n_steps).tolist():
            key, sub = jax.random.split(key)
            x, reveal = _denoise_step(model, x, in_block, jnp.int32(k), sub, temperature,
                                      strategy=strategy, top_k=top_k, greedy=bool(temperature <= 0))
            if record:
                history.append((np.asarray(x[0]), np.asarray(reveal[0])))
    return np.asarray(x[0]), history
