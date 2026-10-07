"""Train Gemma3-1B-IT to do math reasoning on GSM8K with GRPO, from scratch.

Only the Gemma3 model definition and checkpoint loader come from Tunix 0.1.7
(`tunix.models.gemma3`). Everything else -- LoRA adapters, data pipeline,
KV-cache sampler, reward functions, GRPO loss, optimizer loop and evaluation --
is written here in plain JAX + Flax NNX + Optax. The base model is frozen and
only LoRA adapters are trained; everything is stored and computed in bfloat16.

GRPO (Group Relative Policy Optimization, https://arxiv.org/abs/2402.03300):
  1. For each question, sample a *group* of G completions from the policy.
  2. Score every completion with rule-based rewards (format + correctness).
  3. Advantage of a completion = its reward normalized within its group:
       A_i = (r_i - mean(r)) / (std(r) + eps)
     so no value network is needed.
  4. Maximize the PPO-style clipped objective with a KL penalty to a frozen
     reference model:
       L = -E_t[ min(rho_t A, clip(rho_t, 1-eps, 1+eps) A) - beta * KL_t ]
     where rho_t = pi(o_t) / pi_old(o_t) and KL_t uses the k3 estimator
       KL_t = exp(ref_t - logp_t) - (ref_t - logp_t) - 1.
"""

import argparse
import csv
import functools
import json
import math
import os
import re
import sys
import time

import jax
import jax.numpy as jnp
import kagglehub
import numpy as np
import optax
import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import nnx
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--batch_prompts", type=int, default=16, help="questions per step")
parser.add_argument("--num_generations", type=int, default=8, help="group size G")
parser.add_argument("--micro_batch", type=int, default=8, help="sequences per grad-accum micro step")
parser.add_argument("--learning_rate", type=float, default=3e-6)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range")
parser.add_argument("--num_iterations", type=int, default=1, help="policy updates per batch (mu)")
parser.add_argument("--temperature", type=float, default=0.9)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_prompt_len", type=int, default=256)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--eval_size", type=int, default=0, help="test questions to eval on (0 = all 1319)")
parser.add_argument("--eval_batch", type=int, default=128)
parser.add_argument("--skip_initial_eval", action="store_true")
parser.add_argument("--out_dir", type=str, default="artifacts")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

COMPLETION_BUCKET = 128  # pad completions to a multiple of this to limit recompiles
os.makedirs(args.out_dir, exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)  # stream logs even when redirected to a file

# Single-host mesh using the axis names the Tunix Gemma3 sharding config expects.
mesh = jax.make_mesh(
    (jax.device_count(), 1),
    ("fsdp", "tp"),
    axis_types=(jax.sharding.AxisType.Auto,) * 2,
)
jax.sharding.set_mesh(mesh)

# --------------------------------------------------------------------------
# Model + tokenizer (Kaggle checkpoint, loaded with Tunix's Gemma3)
# --------------------------------------------------------------------------
model_dir = kagglehub.model_download("google/gemma-3/flax/gemma3-1b-it")
tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "tokenizer.model"))
BOS_ID, PAD_ID = tokenizer.bos_id(), tokenizer.pad_id()
EOS_IDS = (tokenizer.eos_id(), tokenizer.PieceToId("<end_of_turn>"))

# Everything is stored and computed in bfloat16: weights, activations,
# gradients and optimizer state.
model_config = gemma3_lib.ModelConfig.gemma3_1b_it()
model = gemma3_params.create_model_from_checkpoint(
    os.path.join(model_dir, "gemma3-1b-it"), model_config, mesh=mesh, dtype=jnp.bfloat16
)


# --------------------------------------------------------------------------
# LoRA: y = W x + B A x, with A: d_in -> r, B: r -> d_out and B initialized to 0
# (so training starts exactly at the base model). Only A and B are trained.
# --------------------------------------------------------------------------
class LoRAEinsum(nnx.Module):
    """LoRA adapter around a Tunix `Einsum` such as 'BTD,NDH->BTNH'.

    A maps the contracted input dims (D here) to rank r; B maps r to the
    weight's output dims (N, H), so the adapter output has the base's shape.
    """

    def __init__(self, base, rank, *, rngs):
        self.base = base
        inputs, out_dims = base.einsum_str.split("->")
        x_dims, w_dims = inputs.split(",")
        contracted = "".join(d for d in w_dims if d in x_dims)
        w_out = "".join(d for d in w_dims if d not in x_dims)
        x_rest = "".join(d for d in x_dims if d not in contracted)
        self.down_str = f"{x_dims},{contracted}r->{x_rest}r"  # e.g. BTD,Dr->BTr
        self.up_str = f"{x_rest}r,r{w_out}->{out_dims}"  # e.g. BTr,rNH->BTNH
        size = dict(zip(w_dims, base.shape))
        in_shape = [size[d] for d in contracted]
        out_shape = [size[d] for d in w_out]
        a = nnx.initializers.he_uniform()(rngs.params(), (math.prod(in_shape), rank), jnp.bfloat16)
        self.lora_a = nnx.LoRAParam(a.reshape(*in_shape, rank))
        self.lora_b = nnx.LoRAParam(jnp.zeros((rank, *out_shape), jnp.bfloat16))

    @property
    def shape(self):  # Tunix's Attention reads head counts/dims from this
        return self.base.shape

    def __call__(self, x):
        lora = jnp.einsum(self.up_str, jnp.einsum(self.down_str, x, self.lora_a[...]), self.lora_b[...])
        return self.base(x) + lora


def apply_lora(model, rank, rngs):
    """Wraps every projection in every decoder layer with a LoRA adapter."""
    for layer in model.layers:
        attn = layer.attn
        attn.q_einsum = LoRAEinsum(attn.q_einsum, rank, rngs=rngs)
        attn.kv_einsum = LoRAEinsum(attn.kv_einsum, rank, rngs=rngs)
        attn.attn_vec_einsum = LoRAEinsum(attn.attn_vec_einsum, rank, rngs=rngs)
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            lora = nnx.LoRA(
                linear.in_features, rank, linear.out_features,
                base_module=linear, param_dtype=jnp.bfloat16, rngs=rngs,
            )
            setattr(layer.mlp, name, lora)
    return model


model = apply_lora(model, args.lora_rank, nnx.Rngs(args.seed))
graphdef, lora_params, base_params = nnx.split(model, nnx.LoRAParam, ...)
del model
# Reference policy for the KL term = base model = LoRA adapters zeroed.
ref_lora_params = jax.tree.map(jnp.zeros_like, lora_params)
count = lambda tree: sum(x.size for x in jax.tree.leaves(tree))
print(f"LoRA params: {count(lora_params) / 1e6:.1f}M, frozen base params: {count(base_params) / 1e6:.0f}M")

# --------------------------------------------------------------------------
# Data: GSM8K (Kaggle mirror of openai/grade-school-math)
# --------------------------------------------------------------------------
REASONING_START, REASONING_END = "<reasoning>", "</reasoning>"
ANSWER_START, ANSWER_END = "<answer>", "</answer>"
SYSTEM_PROMPT = (
    "You are given a problem. Think about the problem and provide your reasoning. "
    f"Place it between {REASONING_START} and {REASONING_END}. Then, provide the final "
    f"answer (i.e., just one numerical value) between {ANSWER_START} and {ANSWER_END}."
)
# Gemma's chat template (Gemma has no system role, so it goes in the user turn).
PROMPT_TEMPLATE = "<start_of_turn>user\n{system}\n\n{question}<end_of_turn>\n<start_of_turn>model\n"


def load_gsm8k(split):
    data_dir = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    with open(os.path.join(data_dir, f"main_{split}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    examples = []
    for row in rows:
        prompt_ids = [BOS_ID] + tokenizer.EncodeAsIds(
            PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, question=row["question"])
        )
        if len(prompt_ids) > args.max_prompt_len:
            continue
        examples.append({
            "question": row["question"],
            "prompt_ids": prompt_ids,
            "answer": row["answer"].split("####")[-1].strip(),
        })
    print(f"GSM8K {split}: kept {len(examples)}/{len(rows)} examples (prompt <= {args.max_prompt_len} tokens)")
    return examples


train_data = load_gsm8k("train")
test_data = load_gsm8k("test")
if args.eval_size:
    test_data = test_data[: args.eval_size]


def pad_prompts(examples):
    """Left-pad prompts to max_prompt_len so all completions start at the same slot."""
    ids = np.full((len(examples), args.max_prompt_len), PAD_ID, np.int32)
    for i, ex in enumerate(examples):
        ids[i, -len(ex["prompt_ids"]):] = ex["prompt_ids"]
    return ids, ids != PAD_ID


# --------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------
match_format = re.compile(
    rf"^\s*{REASONING_START}.+?{REASONING_END}.*?{ANSWER_START}(.+?){ANSWER_END}\s*$",
    flags=re.MULTILINE | re.DOTALL,
)
match_number = re.compile(r"-?[\d,]*\.?\d+")


def to_number(text):
    found = match_number.findall(text.replace("$", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def score(completion, answer):
    """Returns (reward, is_correct, has_format, lenient_correct)."""
    target = to_number(answer)
    m = match_format.search(completion)
    has_format = m is not None
    is_correct = has_format and to_number(m.group(1)) == target
    # Lenient: last number anywhere in the text (ignores formatting).
    lenient_correct = to_number(completion) == target
    reward = 0.5 * has_format + 2.0 * is_correct
    return reward, is_correct, has_format, lenient_correct


def decode(tokens, mask):
    return tokenizer.DecodeIds([int(t) for t, m in zip(tokens, mask) if m and t not in EOS_IDS])


# --------------------------------------------------------------------------
# Forward pass + sampler with KV cache
# --------------------------------------------------------------------------
def hidden_states(model, tokens, positions, cache, attn_mask):
    """Gemma3 forward pass up to the final norm (no vocab projection).

    Same as `Gemma3.__call__` minus the LM head. In Tunix 0.1.7 `__call__`
    always projects every position onto the 262k-token vocabulary; doing it
    ourselves lets callers project only the positions they need.
    """
    x = model.embedder.encode(tokens)
    new_cache = None if cache is None else {}
    for i, layer in enumerate(model.layers):
        name = f"layer_{i}"
        layer_cache, x = layer(x, positions, None if cache is None else cache[name], attn_mask)
        if cache is not None:
            new_cache[name] = layer_cache
    return model.final_norm(x), new_cache


def last_token_logits(model, hidden):
    return model.embedder.decode(hidden[:, -1]).astype(jnp.float32)


def init_cache(batch_size, cache_size):
    shape = (batch_size, cache_size, model_config.num_kv_heads, model_config.head_dim)
    return {
        f"layer_{i}": {
            "k": jnp.zeros(shape, jnp.bfloat16),
            "v": jnp.zeros(shape, jnp.bfloat16),
            "end_index": jnp.zeros((batch_size,), jnp.int32),
        }
        for i in range(model_config.num_layers)
    }


def sample_token(logits, key, temperature, top_k):
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    top_logits, top_idx = jax.lax.top_k(logits / temperature, top_k)
    choice = jax.random.categorical(key, top_logits, axis=-1)
    return jnp.take_along_axis(top_idx, choice[:, None], axis=-1)[:, 0].astype(jnp.int32)


@functools.partial(jax.jit, static_argnames=("max_new_tokens", "temperature", "top_k"))
def generate(params, prompt_ids, prompt_mask, key, *, max_new_tokens, temperature, top_k):
    """Samples completions. Returns (tokens, mask), both [B, max_new_tokens]."""
    model = nnx.merge(graphdef, *params)
    B, P = prompt_ids.shape
    S = P + max_new_tokens
    cache = init_cache(B, S)

    # Prefill the whole (left-padded) prompt.
    kv_mask = jnp.pad(prompt_mask, ((0, 0), (0, max_new_tokens)))  # valid cache slots
    positions = jnp.maximum(jnp.cumsum(prompt_mask, axis=-1) - 1, 0)
    attn_mask = jnp.tril(jnp.ones((P, S), jnp.bool_))[None] & kv_mask[:, None, :]
    hidden, cache = hidden_states(model, prompt_ids, positions, cache, attn_mask)
    key, subkey = jax.random.split(key)
    token = sample_token(last_token_logits(model, hidden), subkey, temperature, top_k)
    next_pos = prompt_mask.sum(axis=-1)

    def cond(state):
        t, *_, done, _, _, _ = state
        return (t < max_new_tokens) & ~jnp.all(done)

    def body(state):
        t, token, cache, kv_mask, done, out, out_mask, key = state
        out = out.at[:, t].set(jnp.where(done, PAD_ID, token))
        out_mask = out_mask.at[:, t].set(~done)
        done = done | jnp.isin(token, jnp.array(EOS_IDS))
        # Feed the token at cache slot P + t.
        kv_mask = kv_mask.at[:, P + t].set(True)
        hidden, cache = hidden_states(model, token[:, None], (next_pos + t)[:, None], cache, kv_mask[:, None, :])
        key, subkey = jax.random.split(key)
        token = sample_token(last_token_logits(model, hidden), subkey, temperature, top_k)
        return t + 1, token, cache, kv_mask, done, out, out_mask, key

    state = (
        jnp.int32(0), token, cache, kv_mask, jnp.zeros((B,), jnp.bool_),
        jnp.full((B, max_new_tokens), PAD_ID, jnp.int32), jnp.zeros((B, max_new_tokens), jnp.bool_), key,
    )
    state = jax.lax.while_loop(cond, body, state)
    return state[5], state[6]


# --------------------------------------------------------------------------
# Log-probs and GRPO loss
# --------------------------------------------------------------------------
def completion_logprobs(model, tokens, mask):
    """Per-token log pi(o_t | prompt, o_<t) for the completion part of `tokens`.

    tokens/mask: [B, P + C] (left-padded prompt followed by right-padded completion).
    Returns [B, C]. Only the C completion positions are projected onto the vocabulary.
    """
    B, L = tokens.shape
    P = args.max_prompt_len
    C = L - P
    positions = jnp.maximum(jnp.cumsum(mask, axis=-1) - 1, 0)
    attn_mask = jnp.tril(jnp.ones((L, L), jnp.bool_))[None] & mask[:, None, :]
    hidden, _ = hidden_states(model, tokens, positions, None, attn_mask)
    hidden = hidden[:, P - 1 : -1]  # hidden state at t predicts token t + 1
    targets = tokens[:, P:]
    # Logits in fp32 (like Tunix's `compute_final_logits`): a bf16 log-softmax
    # over 262k entries is too coarse for the policy ratio.
    embedding = model.embedder.input_embedding[...]
    logits = jnp.dot(hidden, embedding.T, preferred_element_type=jnp.float32)
    tgt_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return tgt_logits - jax.nn.logsumexp(logits, axis=-1)


@jax.jit
def frozen_logprobs(params, batch):
    """Log-probs under fixed params (reference / old policy), per micro batch."""
    def micro(mb):
        model = nnx.merge(graphdef, *params)
        return completion_logprobs(model, mb["tokens"], mb["mask"])

    return jax.lax.map(micro, batch)


def grpo_loss(lora_params, base_params, mb, use_old_logp):
    model = nnx.merge(graphdef, lora_params, base_params)
    logp = completion_logprobs(model, mb["tokens"], mb["mask"])
    old_logp = mb["old_logp"] if use_old_logp else jax.lax.stop_gradient(logp)
    comp_mask = mb["mask"][:, args.max_prompt_len :].astype(jnp.float32)
    adv = mb["advantages"][:, None]

    ratio = jnp.exp(logp - old_logp)
    clipped = jnp.clip(ratio, 1 - args.epsilon, 1 + args.epsilon)
    pg_loss = -jnp.minimum(ratio * adv, clipped * adv)
    log_ratio_ref = mb["ref_logp"] - logp
    kl = jnp.exp(log_ratio_ref) - log_ratio_ref - 1
    per_token = pg_loss + args.beta * kl

    # Average over tokens of each sequence, then over sequences (as in GRPO paper).
    seq_mean = lambda x: ((x * comp_mask).sum(-1) / jnp.maximum(comp_mask.sum(-1), 1)).mean()
    loss = seq_mean(per_token)
    clip_frac = seq_mean((jnp.abs(ratio - 1) > args.epsilon).astype(jnp.float32))
    return loss, {"loss": loss, "kl": seq_mean(kl), "clip_frac": clip_frac}


schedule = optax.warmup_cosine_decay_schedule(
    0.0, args.learning_rate, args.warmup_steps, args.num_steps * args.num_iterations, end_value=0.0
)
optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adamw(schedule, b1=0.9, b2=0.99, weight_decay=0.0),
)
opt_state = optimizer.init(lora_params)


def stochastic_round(x, key):
    """Rounds fp32 -> bf16 up or down at random, with probability proportional
    to proximity, so the rounding is unbiased: E[round(x)] = x.

    bf16 is the top 16 bits of fp32: add random noise to the 16 bits that get
    dropped, then truncate.
    """
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    noise = jax.random.bits(key, x.shape, jnp.uint32) >> 16
    bits = (bits + noise) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def apply_updates_stochastic(params, updates, key):
    """`optax.apply_updates` for bf16 params.

    Adam steps are ~lr in size, often far below the bf16 spacing of a weight
    (~0.4% of its magnitude), so round-to-nearest would silently drop them.
    Stochastic rounding keeps them on average.
    """
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))
    add = lambda p, u, k: stochastic_round(p.astype(jnp.float32) + u.astype(jnp.float32), k)
    return jax.tree.map(add, params, updates, keys)


@functools.partial(jax.jit, donate_argnames=("lora_params", "opt_state"), static_argnames=("use_old_logp",))
def train_step(lora_params, opt_state, base_params, batch, key, use_old_logp):
    """One optimizer step; `batch` leaves have shape [num_micro, micro_batch, ...]."""
    grad_fn = jax.grad(grpo_loss, has_aux=True)

    def micro_step(grads, mb):
        g, metrics = grad_fn(lora_params, base_params, mb, use_old_logp)
        return jax.tree.map(jnp.add, grads, g), metrics

    zeros = jax.tree.map(jnp.zeros_like, lora_params)
    grads, metrics = jax.lax.scan(micro_step, zeros, batch)
    grads = jax.tree.map(lambda g: g / batch["tokens"].shape[0], grads)
    updates, opt_state = optimizer.update(grads, opt_state, lora_params)
    lora_params = apply_updates_stochastic(lora_params, updates, key)
    metrics = jax.tree.map(jnp.mean, metrics)
    metrics["grad_norm"] = optax.global_norm(grads)
    return lora_params, opt_state, metrics


# --------------------------------------------------------------------------
# Evaluation (greedy decoding on the GSM8K test set)
# --------------------------------------------------------------------------
def evaluate(lora_params, step):
    params = (lora_params, base_params)
    stats = np.zeros(3)
    n = len(test_data)
    t0 = time.time()
    for start in range(0, n, args.eval_batch):
        chunk = test_data[start : start + args.eval_batch]
        # Pad the last batch to a fixed shape to avoid recompilation.
        padded = chunk + [chunk[-1]] * (args.eval_batch - len(chunk))
        ids, mask = pad_prompts(padded)
        out, out_mask = generate(
            params, ids, mask, jax.random.key(0),
            max_new_tokens=args.max_new_tokens, temperature=0.0, top_k=1,
        )
        out, out_mask = jax.device_get((out, out_mask))
        for i, ex in enumerate(chunk):
            _, correct, fmt, lenient = score(decode(out[i], out_mask[i]), ex["answer"])
            stats += (correct, fmt, lenient)
    acc, fmt, lenient = stats / n
    print(
        f"[eval step {step}] accuracy {acc:.2%} | format {fmt:.2%} | "
        f"lenient accuracy {lenient:.2%} | {n} questions in {time.time() - t0:.0f}s"
    )
    return {"step": step, "eval_accuracy": acc, "eval_format": fmt, "eval_lenient_accuracy": lenient}


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def round_up(x, m):
    return (x + m - 1) // m * m


def build_batch(prompt_ids, prompt_mask, comp_tokens, comp_mask, advantages):
    """Concatenate prompt + completion and reshape into micro batches."""
    # Trim the completion buffer to the longest completion (bucketed to limit recompiles).
    C = min(round_up(max(int(comp_mask.sum(-1).max()), 1), COMPLETION_BUCKET), args.max_new_tokens)
    tokens = np.concatenate([prompt_ids, comp_tokens[:, :C]], axis=1)
    mask = np.concatenate([prompt_mask, comp_mask[:, :C]], axis=1)
    num_micro = tokens.shape[0] // args.micro_batch
    split = lambda x: x.reshape(num_micro, args.micro_batch, *x.shape[1:])
    return {"tokens": split(tokens), "mask": split(mask), "advantages": split(advantages.astype(np.float32))}


rng = np.random.default_rng(args.seed)
key = jax.random.key(args.seed)
order = rng.permutation(len(train_data))
cursor = 0
G = args.num_generations
assert (args.batch_prompts * G) % args.micro_batch == 0

history = []
log_path = os.path.join(args.out_dir, "metrics.jsonl")
log_file = open(log_path, "w")


def log(record):
    history.append(record)
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()


if not args.skip_initial_eval:
    log(evaluate(lora_params, 0))

for step in range(1, args.num_steps + 1):
    t0 = time.time()
    if cursor + args.batch_prompts > len(order):
        order, cursor = rng.permutation(len(train_data)), 0
    examples = [train_data[i] for i in order[cursor : cursor + args.batch_prompts]]
    cursor += args.batch_prompts

    # 1) Rollout: G samples per question (repeat each prompt G times).
    group = [ex for ex in examples for _ in range(G)]
    prompt_ids, prompt_mask = pad_prompts(group)
    key, subkey = jax.random.split(key)
    comp_tokens, comp_mask = generate(
        (lora_params, base_params), prompt_ids, prompt_mask, subkey,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
    )
    comp_tokens, comp_mask = jax.device_get((comp_tokens, comp_mask))
    t_gen = time.time() - t0

    # 2) Rewards and group-relative advantages.
    texts = [decode(comp_tokens[i], comp_mask[i]) for i in range(len(group))]
    scores = np.array([score(text, ex["answer"]) for text, ex in zip(texts, group)], dtype=np.float32)
    rewards = scores[:, 0].reshape(-1, G)
    advantages = (rewards - rewards.mean(1, keepdims=True)) / (rewards.std(1, keepdims=True) + 1e-4)

    # 3) Policy update(s).
    batch = build_batch(prompt_ids, prompt_mask, comp_tokens, comp_mask, advantages.reshape(-1))
    batch["ref_logp"] = frozen_logprobs((ref_lora_params, base_params), batch)
    use_old_logp = args.num_iterations > 1
    if use_old_logp:
        batch["old_logp"] = frozen_logprobs((lora_params, base_params), batch)
    for _ in range(args.num_iterations):
        key, subkey = jax.random.split(key)
        lora_params, opt_state, metrics = train_step(lora_params, opt_state, base_params, batch, subkey, use_old_logp)
    metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
    t_total = time.time() - t0

    record = {
        "step": step,
        "reward": float(rewards.mean()),
        "accuracy": float(scores[:, 1].mean()),
        "format": float(scores[:, 2].mean()),
        "completion_len": float(comp_mask.sum(-1).mean()),
        "lr": float(schedule(step * args.num_iterations)),
        **metrics,
    }
    log(record)
    print(
        f"step {step:4d} | reward {record['reward']:.3f} | acc {record['accuracy']:.2%} | "
        f"format {record['format']:.2%} | len {record['completion_len']:.0f} | "
        f"kl {metrics['kl']:.4f} | loss {metrics['loss']:+.4f} | gnorm {metrics['grad_norm']:.3f} | "
        f"gen {t_gen:.1f}s total {t_total:.1f}s"
    )
    if step % 25 == 0:
        print(f"--- sample (answer {group[0]['answer']}) ---\n{texts[0]}\n---")
    if step % args.eval_every == 0 or step == args.num_steps:
        log(evaluate(lora_params, step))

log_file.close()

# Save the trained LoRA adapters (bf16) with Orbax.
ckpt_dir = os.path.abspath(os.path.join(args.out_dir, "gemma3_1b_grpo_lora"))
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir, lora_params, force=True)
checkpointer.wait_until_finished()
print(f"saved checkpoint to {ckpt_dir}, metrics to {log_path}")

# Plot training reward/accuracy (smoothed) and test accuracy.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_hist = [r for r in history if "reward" in r]
eval_hist = [r for r in history if "eval_accuracy" in r]
smooth = lambda x, w=20: np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else np.array(x)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes[:2], ["reward", "accuracy"]):
    y = smooth([r[k] for r in train_hist])
    ax.plot(np.arange(len(y)) + len(train_hist) - len(y) + 1, y)
    ax.set(title=f"train {k} (20-step avg)", xlabel="step")
for k in ["eval_accuracy", "eval_format", "eval_lenient_accuracy"]:
    axes[2].plot([r["step"] for r in eval_hist], [r[k] for r in eval_hist], marker="o", label=k[5:])
axes[2].set(title="GSM8K test (greedy)", xlabel="step")
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=120)
