import argparse
import json
import math
import os
import time

# Default to CPU backend when no hardware accelerator device is present,
# avoiding noisy libtpu initialization warnings on CPU-only machines.
if (
    "JAX_PLATFORMS" not in os.environ
    and not os.path.exists("/dev/accel0")
    and not os.path.exists("/dev/vfio/0")
    and not os.path.exists("/dev/nvidia0")
):
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

import flax.nnx as nnx
import grain.python as pygrain
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import tiktoken


def load_tokenizer() -> tiktoken.Encoding:
    """Return the GPT-2 tiktoken tokenizer extended with a [MASK] special token."""
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_dllm",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={
            **base._special_tokens,
            "[MASK]": base.n_vocab,
        },
    )


# Create a `Mesh` object representing device arrangement (as in JAX_for_LLM_pretraining)
if jax.device_count() == 8:
    mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ("batch", "model"))
elif jax.device_count() == 1:
    mesh = Mesh(mesh_utils.create_device_mesh((1, 1)), ("batch", "model"))
else:
    mesh = Mesh(
        mesh_utils.create_device_mesh((jax.device_count(), 1)), ("batch", "model")
    )

if hasattr(jax, "set_mesh"):
    jax.set_mesh(mesh)

# Tokenizer (GPT-2 from tiktoken extended with [MASK] token for diffusion)
tokenizer = load_tokenizer()
vocab_size = tokenizer.n_vocab
mask_token_id = tokenizer.encode_single_token("[MASK]")
eot_token_id = tokenizer.encode_single_token("<|endoftext|>")

# Hyperparameters (aligned with miniGPT tutorial + micro-dllm)
maxlen = 256
block_size = maxlen
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
num_transformer_blocks = 4
top_k = 10
num_epochs = 2
steps_per_epoch = 8200

T = 100  # diffusion steps
learning_rate = 1e-3
batch_size = 16 if jax.default_backend() == "cpu" else 64
max_iters = steps_per_epoch * num_epochs
eval_interval = 200
eval_iters = 10 if jax.default_backend() == "cpu" else 50
save_interval = steps_per_epoch

checkpoint_path = "artifacts/models/minigpt_tinystories_ckpt"
loss_curve_path = "artifacts/media/loss_curves.png"
stories_path = "TinyStories-train.txt"

_np_rng = np.random.default_rng(1337)
_jax_rng = jax.random.PRNGKey(1337)


def next_rng_key():
    global _jax_rng
    _jax_rng, subkey = jax.random.split(_jax_rng)
    return subkey


def encode(s: str) -> list[int]:
    return tokenizer.encode(s, allowed_special={"<|endoftext|>", "[MASK]"})


def decode(l: list[int]) -> str:
    return tokenizer.decode(l)


# Diffusion Schedule
def survival_prob(t: float | int) -> float:
    # cosine schedule (better than linear)
    return math.cos((t / T) * math.pi / 2) ** 2


# Precompute survival probabilities for t = 0..T
_SURVIVAL_PROBS = np.array([survival_prob(t) for t in range(T + 1)], dtype=np.float32)

# Full TinyStories dataset state (loaded on demand)
_loaded_stories_path: str | None = None
_all_stories: list[str] = []
train_stories: list[str] = []
val_stories: list[str] = []
data: np.ndarray = np.empty((0,), dtype=np.int32)
train_data: np.ndarray = np.empty((0,), dtype=np.int32)
val_data: np.ndarray = np.empty((0,), dtype=np.int32)


def _ensure_data_loaded(file_path: str | None = None) -> None:
    global _loaded_stories_path, _all_stories, train_stories, val_stories
    global data, train_data, val_data
    target_path = stories_path if file_path is None else file_path
    if _loaded_stories_path == target_path and len(_all_stories) > 0:
        return

    print(f"Loading entire dataset from {target_path}...")
    t0 = time.time()
    with open(target_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    stories = text.split("<|endoftext|>")
    while stories and not stories[-1].strip():
        stories.pop()

    _all_stories = stories
    n_train = int(0.9 * len(_all_stories))
    train_stories = _all_stories[:n_train]
    val_stories = _all_stories[n_train:]
    _loaded_stories_path = target_path

    # Pre-tokenize a contiguous token buffer from train/val stories for prompt sampling & compatibility
    sample_train_text = "<|endoftext|>".join(train_stories[:200]) + "<|endoftext|>"
    sample_val_text = "<|endoftext|>".join(val_stories[:100]) + "<|endoftext|>"
    train_data = np.asarray(encode(sample_train_text), dtype=np.int32)
    val_data = np.asarray(encode(sample_val_text), dtype=np.int32)
    data = np.concatenate([train_data, val_data], axis=0)
    print(
        f"Loaded {len(_all_stories):,} stories "
        f"(train: {len(train_stories):,}, val: {len(val_stories):,}) "
        f"in {time.time() - t0:.2f}s"
    )


class TextDataset:
    """Dataset for TinyStories text (from JAX_for_LLM_pretraining)."""

    def __init__(self, stories_data: list[str], maxlen: int = maxlen):
        self.data = stories_data
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray:
        n = len(self.data)
        cur = idx % n
        token_ids = tokenizer.encode(
            self.data[cur], allowed_special={"<|endoftext|>", "[MASK]"}
        )
        while len(token_ids) < self.maxlen:
            token_ids.append(eot_token_id)
            cur = (cur + 1) % n
            token_ids.extend(
                tokenizer.encode(
                    self.data[cur], allowed_special={"<|endoftext|>", "[MASK]"}
                )
            )
        return np.asarray(token_ids[: self.maxlen], dtype=np.int32)


def load_and_preprocess_data(
    file_path: str = stories_path,
    batch_size: int = batch_size,
    maxlen: int = maxlen,
    max_stories: int | None = None,
    num_epochs: int = num_epochs,
    shuffle: bool = False,
    seed: int = 42,
    stories_subset: list[str] | None = None,
):
    """Loads and preprocesses the entire TinyStories dataset using Grain (as in JAX_for_LLM_pretraining)."""
    if stories_subset is not None:
        stories = stories_subset
    else:
        _ensure_data_loaded(file_path)
        stories = _all_stories

    if max_stories is not None:
        stories = stories[:max_stories]

    dataset = TextDataset(stories, maxlen=maxlen)
    no_shard = (
        pygrain.NoSharding()
        if hasattr(pygrain, "NoSharding")
        else pygrain.NoShard()
    )
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=no_shard,
        num_epochs=num_epochs,
    )
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=0,
    )
    batches_per_epoch = len(dataset) // batch_size
    return dataloader, batches_per_epoch


def corrupt_batch(x0_np: np.ndarray, t_value: int | None = None):
    """Applies forward diffusion masking to a batch of clean token IDs x0."""
    x0 = np.asarray(x0_np, dtype=np.int32)
    if x0.ndim == 2 and x0.shape[0] == block_size and x0.shape[1] != block_size:
        x0 = x0.T
    bsz = x0.shape[0]

    if t_value is None:
        t = _np_rng.integers(1, T + 1, size=(bsz,), dtype=np.int32)
        a_t = _SURVIVAL_PROBS[t][:, None]
    else:
        t_val = int(t_value)
        t = np.full((bsz,), t_val, dtype=np.int32)
        a_t = survival_prob(t_val)

    token_mask = _np_rng.random((bsz, block_size), dtype=np.float32) > a_t
    xt = x0.copy()
    xt[token_mask] = mask_token_id

    data_sharding = NamedSharding(mesh, P("batch", None))
    vec_sharding = NamedSharding(mesh, P("batch"))
    return (
        jax.device_put(jnp.asarray(xt, dtype=jnp.int32), data_sharding),
        jax.device_put(jnp.asarray(x0, dtype=jnp.int32), data_sharding),
        jax.device_put(jnp.asarray(token_mask, dtype=jnp.bool_), data_sharding),
        jax.device_put(jnp.asarray(t, dtype=jnp.int32), vec_sharding),
    )


def _sample_clean_stories_batch(split: str, bsz: int) -> np.ndarray:
    _ensure_data_loaded()
    stories_split = train_stories if split == "train" else val_stories
    indices = _np_rng.integers(0, len(stories_split), size=(bsz,))
    rows = []
    for idx in indices:
        toks = encode(stories_split[int(idx)])
        # Pack additional consecutive stories if shorter than block_size
        cur = int(idx)
        while len(toks) < block_size:
            toks.append(eot_token_id)
            cur = (cur + 1) % len(stories_split)
            toks.extend(encode(stories_split[cur]))
        rows.append(np.asarray(toks[:block_size], dtype=np.int32))
    return np.stack(rows, axis=0)


# Batch Loader
def get_batch(split: str, current_batch_size: int | None = None):
    bsz = batch_size if current_batch_size is None else current_batch_size
    x0 = _sample_clean_stories_batch(split, bsz)
    return corrupt_batch(x0, t_value=None)


def get_batch_at_t(split: str, t_value: int, current_batch_size: int | None = None):
    bsz = batch_size if current_batch_size is None else current_batch_size
    x0 = _sample_clean_stories_batch(split, bsz)
    return corrupt_batch(x0, t_value=t_value)


# Define a triangular mask for causal attention with `jax.numpy.tril` and `jax.numpy.ones`.
def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


class TransformerBlock(nnx.Module):
    """A single Transformer block (from JAX_for_LLM_pretraining).

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
        rate: float = 0.1,
    ):
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        # Specifies tensor sharding (depending on the mesh configuration)
        # where we shard the weights across devices for parallel computation.
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), P(None, "model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )
        # The first dropout with `flax.nnx.Dropout`.
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        # First layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(), P("model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )
        # The first linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), P(None, "model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )
        # The second linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), P(None, "model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )
        # The second dropout with `flax.nnx.Dropout`.
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        # Second layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(), P("model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )

    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs, training: bool = False, mask=None):
        # Pre-LayerNorm before bidirectional Multi-Head Attention to preserve sequence variance.
        norm1 = self.layer_norm1(inputs)
        attention_output = self.mha(
            inputs_q=norm1,
            mask=mask,
            decode=False,
        )
        # Apply the first dropout.
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output

        # The feed-forward network with Pre-LayerNorm.
        norm2 = self.layer_norm2(out1)
        ffn_output = self.linear1(norm2)
        # Apply the ReLU activation with `flax.nnx.relu`.
        ffn_output = nnx.relu(ffn_output)
        # Apply the second linear transformation.
        ffn_output = self.linear2(ffn_output)
        # Apply the second dropout and residual connection.
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return out1 + ffn_output


class TokenAndPositionEmbedding(nnx.Module):
    """Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        # Initialize token embeddings (using `flax.nnx.Embed`).
        self.token_emb = nnx.Embed(
            num_embeddings=vocab_size, features=embed_dim, rngs=rngs
        )
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(
            num_embeddings=maxlen, features=embed_dim, rngs=rngs
        )
        self.embed_scale = math.sqrt(embed_dim)

    def __call__(self, x):
        # Generate a sequence of positions for the input tokens.
        positions = jnp.arange(0, x.shape[1])[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x)
        # Combine token and positional embeddings scaled by sqrt(embed_dim).
        return (token_embedding + position_embedding) * self.embed_scale


class MiniGPT(nnx.Module):
    """A miniGPT transformer model (from JAX_for_LLM_pretraining) adapted for
    text diffusion with timestep conditioning, inherits from `flax.nnx.Module`.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimensionality of the feed-forward network.
        num_transformer_blocks (int): Number of transformer blocks.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    def __init__(
        self,
        maxlen: int = maxlen,
        vocab_size: int = vocab_size,
        embed_dim: int = embed_dim,
        num_heads: int = num_heads,
        feed_forward_dim: int = feed_forward_dim,
        num_transformer_blocks: int = num_transformer_blocks,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        # Initialize the `TokenAndPositionEmbedding` that combines token and positional embeddings.
        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )
        # Timestep conditioning embedding for diffusion steps t in [0..T].
        self.timestep_emb = nnx.Embed(
            num_embeddings=T + 1, features=embed_dim, rngs=rngs
        )
        # Create a list of `TransformerBlock` instances.
        blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        self.transformer_blocks = (
            nnx.List(blocks) if hasattr(nnx, "List") else blocks
        )
        # Initialize the output `flax.nnx.Linear` layer producing logits over the vocabulary.
        self.output_layer = nnx.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), P(None, "model"), mesh=mesh
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(), P("model"), mesh=mesh
            ),
            rngs=rngs,
        )

    def __call__(self, inputs, t=None, targets=None, mask=None, training: bool = False):
        x = self.embedding_layer(inputs)
        if t is not None:
            x = x + self.timestep_emb(t)[:, None, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=None)
        logits = self.output_layer(x)

        loss = None
        if targets is not None:
            per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets
            )
            if mask is not None:
                mask_f = mask.astype(jnp.float32)
                masked_count = jnp.sum(mask_f)
                loss = jnp.where(
                    masked_count > 0,
                    jnp.sum(per_token_loss * mask_f) / jnp.maximum(masked_count, 1.0),
                    jnp.mean(per_token_loss),
                )
            else:
                loss = jnp.mean(per_token_loss)

        return logits, loss


# Alias Model to MiniGPT for compatibility with micro-dllm's inference.py
Model = MiniGPT


# Creates the miniGPT model with 4 transformer blocks (as in JAX_for_LLM_pretraining).
def create_model(rngs: nnx.Rngs) -> MiniGPT:
    return MiniGPT(
        maxlen=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
        rngs=rngs,
    )


# Defines the loss function using `optax.softmax_cross_entropy_with_integer_labels`.
def loss_fn(model: MiniGPT, batch):
    xt, x0, mask, t = batch
    logits, loss = model(xt, t=t, targets=x0, mask=mask, training=True)
    return loss, logits


# Define the training step with the `flax.nnx.jit` transformation decorator.
@nnx.jit
def train_step(
    model: MiniGPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _eval_masked_step(model: MiniGPT, xb, yb, mb, tb):
    logits, loss = model(xb, t=tb, targets=yb, mask=mb, training=False)
    pred = jnp.argmax(logits, axis=-1)
    mb_i = mb.astype(jnp.int32)
    correct = jnp.sum((pred == yb).astype(jnp.int32) * mb_i)
    total = jnp.sum(mb_i)
    return loss, correct, total


@nnx.jit
def _eval_entropy_step(model: MiniGPT, xb, mb, tb):
    logits, _ = model(xb, t=tb, training=False)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    token_entropy = -jnp.sum(probs * log_probs, axis=-1)
    mb_f = mb.astype(jnp.float32)
    return jnp.sum(token_entropy * mb_f), jnp.sum(mb_f)


@nnx.jit
def _reverse_diffusion_step(
    model: MiniGPT,
    x: jax.Array,
    t_tensor: jax.Array,
    fixed_mask: jax.Array,
    k_remask: jax.Array,
    rng_key: jax.Array,
    temperature: jax.Array,
):
    logits, _ = model(x, t=t_tensor, training=False)

    greedy_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    greedy_probs = jax.nn.softmax(logits, axis=-1)
    greedy_conf = jnp.take_along_axis(
        greedy_probs, greedy_tokens[..., None], axis=-1
    ).squeeze(-1)

    temp_safe = jnp.maximum(temperature, 1e-6)
    scaled_logits = logits / temp_safe
    stoch_tokens = jax.random.categorical(rng_key, scaled_logits, axis=-1).astype(
        jnp.int32
    )
    stoch_probs = jax.nn.softmax(scaled_logits, axis=-1)
    stoch_conf = jnp.take_along_axis(
        stoch_probs, stoch_tokens[..., None], axis=-1
    ).squeeze(-1)

    use_greedy = temperature <= 0.0
    sampled = jnp.where(use_greedy, greedy_tokens, stoch_tokens)
    sampled_conf = jnp.where(use_greedy, greedy_conf, stoch_conf)

    is_masked = (x == mask_token_id) & (~fixed_mask)
    x_next = jnp.where(is_masked, sampled, x)

    # Confidence-based remasking of the k lowest-confidence currently-masked positions
    masked_conf = jnp.where(is_masked, sampled_conf, jnp.inf)
    ranks = jnp.argsort(jnp.argsort(masked_conf, axis=-1), axis=-1)
    remask = (ranks < k_remask) & is_masked
    x_next = jnp.where(remask, mask_token_id, x_next)
    return x_next


@nnx.jit
def _final_denoise_step(model: MiniGPT, x: jax.Array, t0_tensor: jax.Array, fixed_mask: jax.Array):
    logits, _ = model(x, t=t0_tensor, training=False)
    final_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    is_masked = (x == mask_token_id) & (~fixed_mask)
    return jnp.where(is_masked, final_tokens, x)


def save_checkpoint(model: MiniGPT, path: str, step: int, loss_val: float) -> None:
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    state = nnx.state(model, nnx.Param)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(abs_path, args=ocp.args.PyTreeSave(state), force=True)


def load_checkpoint(model: MiniGPT, path: str) -> MiniGPT:
    abs_path = os.path.abspath(path)
    target_state = nnx.state(model, nnx.Param)
    restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=getattr(x, "sharding", None)),
        target_state,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        abs_path,
        args=ocp.args.PyTreeRestore(item=target_state, restore_args=restore_args),
    )
    nnx.update(model, restored_state)
    return model


def estimate_loss(model: MiniGPT, num_batches: int | None = None):
    n_batches = eval_iters if num_batches is None else num_batches
    losses = {}
    for split in ["train", "val"]:
        split_losses = []
        for _ in range(n_batches):
            xb, yb, mb, tb = get_batch(split)
            loss, _, _ = _eval_masked_step(model, xb, yb, mb, tb)
            split_losses.append(float(loss))
        losses[split] = float(np.mean(split_losses))
    return losses


def save_loss_curves(eval_steps, train_losses, val_losses, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping loss curve plot.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(eval_steps, train_losses, label="train masked loss", marker="o", markersize=4)
    plt.plot(eval_steps, val_losses, label="val masked loss", marker="o", markersize=4)
    if eval_steps and max(eval_steps) > steps_per_epoch:
        for ep in range(1, (max(eval_steps) // steps_per_epoch) + 1):
            ep_step = ep * steps_per_epoch
            if ep_step < max(eval_steps):
                plt.axvline(
                    x=ep_step,
                    color="#6b7280",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Epoch {ep} End (step {ep_step})",
                )
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Training and Validation Loss Curves ({num_epochs} Epochs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved loss curves to {output_path}", flush=True)


def evaluate_masked_metrics(model: MiniGPT, split: str = "val", num_batches: int | None = None):
    n_batches = eval_iters if num_batches is None else num_batches
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for _ in range(n_batches):
        xb, yb, mb, tb = get_batch(split)
        loss, correct, total = _eval_masked_step(model, xb, yb, mb, tb)
        total_loss += float(loss)
        total_correct += int(correct)
        total_masked += int(total)

    avg_loss = total_loss / max(1, n_batches)
    masked_acc = total_correct / max(1, total_masked)
    perplexity = math.exp(min(avg_loss, 20.0))

    return {
        "masked_loss": avg_loss,
        "masked_recon_acc": masked_acc,
        "perplexity": perplexity,
    }


def evaluate_entropy_per_timestep(model: MiniGPT, split: str = "val", batches_per_t: int = 1):
    entropies = []
    eval_bsz = min(batch_size, 8) if jax.default_backend() == "cpu" else batch_size

    for t_value in range(1, T + 1):
        entropy_sum = 0.0
        entropy_count = 0.0
        for _ in range(batches_per_t):
            xb, _, mb, tb = get_batch_at_t(split, t_value, current_batch_size=eval_bsz)
            e_sum, e_cnt = _eval_entropy_step(model, xb, mb, tb)
            entropy_sum += float(e_sum)
            entropy_count += float(e_cnt)
        entropies.append(entropy_sum / max(1.0, entropy_count))

    return entropies


def generate_with_trace(model: MiniGPT, prompt_tokens: list[int], gen_len: int = 128, temperature: float = 0.0):
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens cannot be empty")
    if len(prompt_tokens) >= block_size:
        raise ValueError("prompt_tokens length must be < block_size")

    max_gen = block_size - len(prompt_tokens)
    gen_len = max(1, min(gen_len, max_gen))
    total_len = len(prompt_tokens) + gen_len
    gen_slice = slice(len(prompt_tokens), total_len)

    x_np = np.full((1, block_size), mask_token_id, dtype=np.int32)
    x_np[0, : len(prompt_tokens)] = np.asarray(prompt_tokens, dtype=np.int32)
    x = jnp.asarray(x_np)

    fixed_mask_np = np.ones((1, block_size), dtype=bool)
    fixed_mask_np[:, gen_slice] = False
    fixed_mask = jnp.asarray(fixed_mask_np)

    states = [np.asarray(x[0, gen_slice]).copy()]
    temp_arr = jnp.asarray(temperature, dtype=jnp.float32)
    gen_positions = total_len - len(prompt_tokens)

    for t in reversed(range(1, T + 1)):
        t_tensor = jnp.asarray([t], dtype=jnp.int32)
        if t > 1:
            next_mask_ratio = 1.0 - survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
        else:
            k = 0
        k_arr = jnp.asarray(k, dtype=jnp.int32)
        x = _reverse_diffusion_step(
            model, x, t_tensor, fixed_mask, k_arr, next_rng_key(), temp_arr
        )
        states.append(np.asarray(x[0, gen_slice]).copy())

    t0 = jnp.asarray([0], dtype=jnp.int32)
    x = _final_denoise_step(model, x, t0, fixed_mask)
    states.append(np.asarray(x[0, gen_slice]).copy())

    change_rates = []
    for prev_state, next_state in zip(states[:-1], states[1:]):
        change_rate = float(np.mean(next_state != prev_state))
        change_rates.append(change_rate)

    return np.asarray(x[0, gen_slice]).tolist(), change_rates


def evaluate_generation_metrics(
    model: MiniGPT,
    split: str = "val",
    num_samples: int = 4,
    prompt_len: int = 32,
    gen_len: int = 128,
    temperature: float = 0.0,
):
    _ensure_data_loaded()
    needed = prompt_len + gen_len + 1

    all_change_rates = []
    total_bigrams = 0
    unique_bigrams = set()

    for _ in range(num_samples):
        clean_seq = _sample_clean_stories_batch(split, max(1, (needed // block_size) + 1)).reshape(-1)
        prompt_tokens = clean_seq[:prompt_len].tolist()

        gen_tokens, change_rates = generate_with_trace(
            model,
            prompt_tokens=prompt_tokens,
            gen_len=gen_len,
            temperature=temperature,
        )

        all_change_rates.extend(change_rates)

        if len(gen_tokens) >= 2:
            for i in range(len(gen_tokens) - 1):
                bg = (gen_tokens[i], gen_tokens[i + 1])
                unique_bigrams.add(bg)
                total_bigrams += 1

    distinct_2 = len(unique_bigrams) / max(1, total_bigrams)
    avg_change_rate = sum(all_change_rates) / max(1, len(all_change_rates))

    return {
        "reverse_step_token_change_rate": avg_change_rate,
        "distinct_2": distinct_2,
    }


# Reverse Diffusion Sampling
def generate(model: MiniGPT, prompt_len: int = 16, temperature: float = 1.0) -> str:
    _ensure_data_loaded()
    x_np = np.full((1, block_size), mask_token_id, dtype=np.int32)
    x_np[0, :prompt_len] = data[:prompt_len]
    x = jnp.asarray(x_np)

    prompt_mask_np = np.zeros((1, block_size), dtype=bool)
    prompt_mask_np[:, :prompt_len] = True
    prompt_mask = jnp.asarray(prompt_mask_np)

    gen_positions = block_size - prompt_len
    temp_arr = jnp.asarray(temperature, dtype=jnp.float32)

    for t in reversed(range(1, T + 1)):
        t_tensor = jnp.asarray([t], dtype=jnp.int32)
        if t > 1:
            next_mask_ratio = 1.0 - survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
        else:
            k = 0
        k_arr = jnp.asarray(k, dtype=jnp.int32)
        x = _reverse_diffusion_step(
            model, x, t_tensor, prompt_mask, k_arr, next_rng_key(), temp_arr
        )

    # Explicit final denoise at t=0.
    t0 = jnp.asarray([0], dtype=jnp.int32)
    x = _final_denoise_step(model, x, t0, prompt_mask)
    return decode(np.asarray(x[0]).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT text diffusion model with JAX/Flax NNX on the entire TinyStories dataset.")
    parser.add_argument("--stories-path", type=str, default=stories_path)
    parser.add_argument("--max-stories", type=int, default=None, help="Maximum stories to load (default: None = entire dataset).")
    parser.add_argument("--num-epochs", type=int, default=num_epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=steps_per_epoch)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--max-iters", type=int, default=None, help="Max training steps (default: steps_per_epoch * num_epochs).")
    parser.add_argument("--eval-interval", type=int, default=eval_interval)
    parser.add_argument("--eval-iters", type=int, default=eval_iters)
    parser.add_argument("--save-interval", type=int, default=save_interval)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--checkpoint", type=str, default=checkpoint_path)
    parser.add_argument("--loss-curve", type=str, default=loss_curve_path)
    parser.add_argument("--resume", action="store_true", help="Resume from existing Orbax checkpoint.")
    parser.add_argument("--start-step", type=int, default=0, help="Starting step index when resuming.")
    args = parser.parse_args()

    stories_path = args.stories_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    steps_per_epoch = args.steps_per_epoch
    max_iters = (
        steps_per_epoch * num_epochs if args.max_iters is None else args.max_iters
    )
    eval_interval = args.eval_interval
    eval_iters = args.eval_iters
    save_interval = args.save_interval
    learning_rate = args.lr
    checkpoint_path = args.checkpoint
    loss_curve_path = args.loss_curve

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)

    print(f"JAX devices: {jax.devices()} | Mesh: {mesh.shape}")
    print(f"Tokenizer: gpt2 (vocab_size={vocab_size}, mask_token_id={mask_token_id})")

    _ensure_data_loaded(stories_path)
    text_dl, batches_per_epoch = load_and_preprocess_data(
        file_path=stories_path,
        batch_size=batch_size,
        maxlen=maxlen,
        max_stories=args.max_stories,
        num_epochs=num_epochs,
        shuffle=True,
        seed=1337 + args.start_step,
        stories_subset=train_stories if args.max_stories is None else None,
    )
    total_target_steps = (
        batches_per_epoch * num_epochs if max_iters <= 0 else max_iters
    )
    print(
        f"Grain DataLoader ready: {len(train_stories) if args.max_stories is None else args.max_stories:,} training stories | "
        f"num_epochs={num_epochs} | steps_per_epoch={steps_per_epoch:,} | target_steps={total_target_steps:,}"
    )

    model = create_model(rngs=nnx.Rngs(1337 + args.start_step))
    if args.resume and os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        print(f"Resumed model from Orbax checkpoint {checkpoint_path} at step {args.start_step}", flush=True)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    eval_steps = []
    train_loss_curve = []
    val_loss_curve = []
    epoch_step_losses = {ep: [] for ep in range(1, num_epochs + 1)}
    epoch_summaries = {}
    loss_val = 0.0

    dl_iter = iter(text_dl)
    for iter_idx in range(args.start_step, total_target_steps):
        current_epoch = min(num_epochs, (iter_idx // max(1, steps_per_epoch)) + 1)

        if iter_idx % eval_interval == 0:
            losses = estimate_loss(model)
            eval_steps.append(iter_idx)
            train_loss_curve.append(losses["train"])
            val_loss_curve.append(losses["val"])
            print(
                f"eval step {iter_idx} (epoch {current_epoch}/{num_epochs}) | "
                f"train masked loss {losses['train']:.4f} | "
                f"val masked loss {losses['val']:.4f}",
                flush=True,
            )

        try:
            clean_batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(text_dl)
            clean_batch = next(dl_iter)

        batch = corrupt_batch(clean_batch)
        loss = train_step(model, optimizer, metrics, batch)
        loss_val = float(loss)
        epoch_step_losses[current_epoch].append(loss_val)

        if iter_idx % 50 == 0 or iter_idx == total_target_steps - 1:
            print(
                f"epoch {current_epoch}/{num_epochs} | step {iter_idx} | loss {loss_val:.4f}",
                flush=True,
            )

        is_epoch_end = ((iter_idx + 1) % steps_per_epoch == 0) or (
            iter_idx == total_target_steps - 1
        )
        if not is_epoch_end and (iter_idx + 1) % 1000 == 0:
            save_checkpoint(model, checkpoint_path, iter_idx + 1, loss_val)
            print(
                f"saved checkpoint to {checkpoint_path} at step {iter_idx + 1}",
                flush=True,
            )

        if is_epoch_end:
            completed_step = iter_idx + 1
            ep_losses = estimate_loss(model)
            ep_val_metrics = evaluate_masked_metrics(
                model, split="val", num_batches=eval_iters
            )
            ep_batch_losses = epoch_step_losses[current_epoch]
            ep_mean_batch_loss = float(np.mean(ep_batch_losses))
            ep_tail_batch_loss = float(np.mean(ep_batch_losses[-100:]))
            epoch_summaries[current_epoch] = {
                "step": completed_step,
                "mean_batch_loss": ep_mean_batch_loss,
                "tail_100_batch_loss": ep_tail_batch_loss,
                "last_step_loss": loss_val,
                "train_masked_loss": ep_losses["train"],
                "val_masked_loss": ep_losses["val"],
                "val_perplexity": ep_val_metrics["perplexity"],
                "val_masked_recon_acc": ep_val_metrics["masked_recon_acc"],
            }
            if completed_step == total_target_steps:
                eval_steps.append(completed_step)
                train_loss_curve.append(ep_losses["train"])
                val_loss_curve.append(ep_losses["val"])

            print(
                f"\n--- End of Epoch {current_epoch}/{num_epochs} (step {completed_step}) ---\n"
                f"  Epoch mean batch loss:      {ep_mean_batch_loss:.4f}\n"
                f"  Last 100-step batch loss:   {ep_tail_batch_loss:.4f}\n"
                f"  Train masked loss (eval):   {ep_losses['train']:.4f}\n"
                f"  Val masked loss (eval):     {ep_losses['val']:.4f}\n"
                f"  Val perplexity:             {ep_val_metrics['perplexity']:.4f}\n"
                f"  Val masked recon accuracy:  {ep_val_metrics['masked_recon_acc']:.4f}",
                flush=True,
            )
            print("Generating sample...", flush=True)
            print(generate(model, temperature=0.0), flush=True)
            save_checkpoint(model, checkpoint_path, completed_step, loss_val)
            print(
                f"saved checkpoint to {checkpoint_path} at step {completed_step}\n",
                flush=True,
            )

    save_loss_curves(eval_steps, train_loss_curve, val_loss_curve, loss_curve_path)

    if len(epoch_summaries) >= 2:
        e1 = epoch_summaries[1]
        e2 = epoch_summaries[2]
        print("\n=== Epoch 1 vs. Epoch 2 Loss Comparison ===")
        print(
            f"Epoch 1 (step {e1['step']}): "
            f"mean_batch_loss={e1['mean_batch_loss']:.4f}, "
            f"tail100_batch_loss={e1['tail_100_batch_loss']:.4f}, "
            f"train_masked_loss={e1['train_masked_loss']:.4f}, "
            f"val_masked_loss={e1['val_masked_loss']:.4f}, "
            f"val_ppl={e1['val_perplexity']:.4f}, "
            f"val_acc={e1['val_masked_recon_acc']:.4f}"
        )
        print(
            f"Epoch 2 (step {e2['step']}): "
            f"mean_batch_loss={e2['mean_batch_loss']:.4f}, "
            f"tail100_batch_loss={e2['tail_100_batch_loss']:.4f}, "
            f"train_masked_loss={e2['train_masked_loss']:.4f}, "
            f"val_masked_loss={e2['val_masked_loss']:.4f}, "
            f"val_ppl={e2['val_perplexity']:.4f}, "
            f"val_acc={e2['val_masked_recon_acc']:.4f}"
        )
        print(
            f"Delta (Epoch 2 - Epoch 1): "
            f"mean_batch_loss={e2['mean_batch_loss'] - e1['mean_batch_loss']:+.4f}, "
            f"tail100_batch_loss={e2['tail_100_batch_loss'] - e1['tail_100_batch_loss']:+.4f}, "
            f"train_masked_loss={e2['train_masked_loss'] - e1['train_masked_loss']:+.4f}, "
            f"val_masked_loss={e2['val_masked_loss'] - e1['val_masked_loss']:+.4f}, "
            f"val_ppl={e2['val_perplexity'] - e1['val_perplexity']:+.4f}, "
            f"val_acc={e2['val_masked_recon_acc'] - e1['val_masked_recon_acc']:+.4f}"
        )

    final_core = evaluate_masked_metrics(model, split="val", num_batches=eval_iters)
    entropy_by_t = evaluate_entropy_per_timestep(model, split="val", batches_per_t=1)
    final_gen = evaluate_generation_metrics(
        model,
        split="val",
        num_samples=4,
        prompt_len=32,
        gen_len=128,
        temperature=0.0,
    )

    entropy_mean = sum(entropy_by_t) / max(1, len(entropy_by_t))
    entropy_t1 = entropy_by_t[0]
    entropy_tmid = entropy_by_t[(len(entropy_by_t) - 1) // 2]
    entropy_tT = entropy_by_t[-1]

    print("\n=== Final Evaluation Metrics (val) ===")
    print(f"Perplexity: {final_core['perplexity']:.4f}")
    print(f"Masked reconstruction accuracy: {final_core['masked_recon_acc']:.4f}")
    print(
        "Entropy per timestep (masked positions): "
        f"mean={entropy_mean:.4f}, t=1:{entropy_t1:.4f}, "
        f"t={1 + (T - 1) // 2}:{entropy_tmid:.4f}, t={T}:{entropy_tT:.4f}"
    )
    print(
        "Reverse-step token change rate: "
        f"{final_gen['reverse_step_token_change_rate']:.4f}"
    )
    print(f"Distinct-2 diversity (generated region): {final_gen['distinct_2']:.4f}")
