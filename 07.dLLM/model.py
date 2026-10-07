"""MiniGPT in Flax NNX, used as the denoiser of a masked diffusion LM.

The architecture is the one built in `01.miniGPT` (TokenAndPositionEmbedding ->
N x post-LayerNorm TransformerBlock with `nnx.MultiHeadAttention` and a ReLU
feed-forward network -> untied `nnx.Linear` output layer).

The only architectural change needed to turn this autoregressive miniGPT into a
*masked diffusion* language model is to drop the causal mask: the denoiser looks
at the whole (partially masked) sequence at once and predicts the clean token at
every masked position.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)  # frozen -> hashable, so it can live on the module as static metadata
class MiniGPTConfig:
    vocab_size: int = 8192
    maxlen: int = 256  # max sequence length
    embed_dim: int = 256
    num_heads: int = 8
    feed_forward_dim: int = 256
    num_transformer_blocks: int = 4
    dropout_rate: float = 0.0


class TransformerBlock(nnx.Module):
    """A single Transformer block (same as 01.miniGPT, minus the causal mask).

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, decode=False, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6, num_features=embed_dim, rngs=rngs)
        # Feed-forward network.
        self.linear1 = nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=ff_dim, out_features=embed_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6, num_features=embed_dim, rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        # Bidirectional attention: `mask=None` <-- the key difference from the autoregressive
        # miniGPT, which passes `mask=causal_attention_mask(seq_len)` here.
        attention_output = self.mha(inputs_q=inputs, mask=None, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.linear1(out1)
        ffn_output = nnx.relu(ffn_output)
        ffn_output = self.linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return self.layer_norm2(out1 + ffn_output)


class TokenAndPositionEmbedding(nnx.Module):
    """Combines token embeddings with learned positional embeddings.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class MiniGPT(nnx.Module):
    """A miniGPT transformer model, inherits from `flax.nnx.Module`.

    Args:
        cfg (MiniGPTConfig): model hyper-parameters.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    def __init__(self, cfg: MiniGPTConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedding_layer = TokenAndPositionEmbedding(cfg.maxlen, cfg.vocab_size, cfg.embed_dim, rngs=rngs)
        self.transformer_blocks = nnx.List([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.feed_forward_dim, rngs=rngs, rate=cfg.dropout_rate)
            for _ in range(cfg.num_transformer_blocks)
        ])
        # Output layer producing logits over the vocabulary (predicts the clean token x_0).
        self.output_layer = nnx.Linear(in_features=cfg.embed_dim, out_features=cfg.vocab_size, rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        """inputs: (B, T) token ids (may contain the <mask> id). Returns logits (B, T, V)."""
        assert inputs.shape[1] <= self.cfg.maxlen
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        return self.output_layer(x)


def num_params(model: nnx.Module, non_embedding: bool = True) -> int:
    n = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    if non_embedding:
        n -= model.embedding_layer.pos_emb.embedding[...].size
    return n
