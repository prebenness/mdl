"""Free-form RNN (Lan et al. 2022, TACL 10:785-799, arXiv:2111.00600).

A free-form RNN is a directed graph where each unit has a per-unit
activation function and weighted connections to other units.  Connections
are either "forward" (within the current timestep, respecting topological
order) or "recurrent" (from the previous timestep).

Output interpretation follows Lan et al. (2022, Section 3.4): zero
negative output values, then normalize to a probability distribution.
For gradient-descent training (Abudy et al. 2025, Exp. 3), softmax is
used instead.

This module provides:
  - FreeFormTopology: immutable description of the graph structure
  - freeform_forward: deterministic forward pass with explicit weights
  - GumbelSoftmaxFreeFormRNN: Flax module for differentiable MDL training
"""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
from dataclasses import dataclass
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

# Encoding cost in bits (Lan et al. 2022, Section 3.3)
ACTIVATION_BITS: dict[str, int] = {
    "linear": 0,
    "square": 2,
    "relu": 4,
    "sigmoid": 4,
    "tanh": 4,
    "floor": 4,
    "unsigned_step": 8,
}


_RELU_CAP = 1e6  # prevent float32 overflow in recurrent scan backward pass


def _apply_activation(x, name: str):
    """Apply a named activation function element-wise."""
    if name == "linear":
        return x
    elif name == "relu":
        # Cap at _RELU_CAP to prevent NaN gradients in lax.scan backward
        # pass: large recurrent weights (e.g. 10) cause counter activations
        # to overflow float32 after ~40 timesteps.  The clipped gradient is
        # 0 at the cap, halting the exponential Jacobian product that would
        # otherwise produce inf, then 0 * inf = NaN.  The cap is far above
        # any activation reached by correct golden networks (counters ≤ n).
        return jnp.minimum(jax.nn.relu(x), _RELU_CAP)
    elif name == "sigmoid":
        return jax.nn.sigmoid(x)
    elif name == "tanh":
        return jnp.tanh(x)
    elif name == "square":
        return x * x
    elif name == "floor":
        return jnp.floor(x)
    elif name == "unsigned_step":
        # 0 for x <= 0, 1 for x > 0
        return (x > 0).astype(x.dtype)
    else:
        raise ValueError(f"Unknown activation: {name!r}")


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreeFormTopology:
    """Directed-graph topology for a free-form RNN (weights not included).

    Units are indexed 0..n_units-1.  Forward connections must go from a
    lower-index unit to a higher-index unit (topological order).

    Attributes:
        n_units: total number of units (input + hidden + output).
        activations: per-unit activation function name.
        connections: each entry is (src_type, src_idx, dst_idx) where
            src_type is "input" (src_idx = input dimension),
            "forward" (src_idx = unit index, must be < dst_idx), or
            "recurrent" (src_idx = unit index, from previous timestep).
        biased_units: frozenset of unit indices that have a bias.
        output_units: tuple of unit indices whose activations become the
            output logits.  len(output_units) == output_size.
        input_size: one-hot input dimension (vocabulary size).
        output_size: output dimension (vocabulary size).
    """
    n_units: int
    activations: tuple
    connections: tuple
    biased_units: frozenset
    output_units: tuple
    input_size: int
    output_size: int

    def __post_init__(self):
        assert len(self.activations) == self.n_units
        assert len(self.output_units) == self.output_size
        for src_type, src_idx, dst_idx in self.connections:
            assert src_type in ("input", "forward", "recurrent")
            if src_type == "forward":
                assert src_idx < dst_idx, (
                    f"Forward {src_idx}->{dst_idx} violates topological order"
                )

    @property
    def n_connection_weights(self) -> int:
        return len(self.connections)

    @property
    def n_bias_weights(self) -> int:
        return len(self.biased_units)

    @property
    def n_weights(self) -> int:
        """Total learnable scalar parameters."""
        return self.n_connection_weights + self.n_bias_weights

    def sorted_biased_units(self) -> list[int]:
        """Biased unit indices in sorted order (defines bias weight layout)."""
        return sorted(self.biased_units)


# ---------------------------------------------------------------------------
# Forward pass (deterministic, explicit weights)
# ---------------------------------------------------------------------------

def freeform_forward(topology: FreeFormTopology, weights, x):
    """Deterministic forward pass through a free-form RNN.

    Args:
        topology: graph structure.
        weights: float array (n_weights,) of connection weights and biases.
            Layout: connections first (in topology.connections order),
            then biases (in sorted biased_units order).
        x: int32 (batch, seq_len) input token indices.

    Returns:
        logits: float32 (batch, seq_len, output_size) raw output-unit
            activations.  Caller applies softmax or zero-neg normalization.
    """
    B, T = x.shape
    N = topology.n_units
    I = topology.input_size

    conn_w = weights[:topology.n_connection_weights]
    bias_w = weights[topology.n_connection_weights:]

    # Bias lookup: unit_idx -> bias value
    sorted_biased = topology.sorted_biased_units()
    bias_of = {}
    for i, uid in enumerate(sorted_biased):
        bias_of[uid] = bias_w[i]

    # Group connections by destination unit for efficient processing
    # Each entry: (conn_index, src_type, src_idx)
    unit_conns: list[list[tuple[int, str, int]]] = [[] for _ in range(N)]
    for ci, (src_type, src_idx, dst_idx) in enumerate(topology.connections):
        unit_conns[dst_idx].append((ci, src_type, src_idx))

    x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

    def step(prev_acts, x_t):
        """Process one timestep.  prev_acts: (B, N), x_t: (B, I)."""
        curr_acts = jnp.zeros((B, N))
        for uid in range(N):
            pre = jnp.zeros((B,))
            if uid in bias_of:
                pre = pre + bias_of[uid]
            for ci, src_type, src_idx in unit_conns[uid]:
                w = conn_w[ci]
                if src_type == "input":
                    pre = pre + w * x_t[:, src_idx]
                elif src_type == "forward":
                    pre = pre + w * curr_acts[:, src_idx]
                elif src_type == "recurrent":
                    pre = pre + w * prev_acts[:, src_idx]
            act_val = _apply_activation(pre, topology.activations[uid])
            curr_acts = curr_acts.at[:, uid].set(act_val)
        return curr_acts, curr_acts

    h0 = jnp.zeros((B, N))
    x_seq = jnp.transpose(x_onehot, (1, 0, 2))  # (T, B, I)
    _, all_acts = jax.lax.scan(step, h0, x_seq)
    # all_acts: (T, B, N) -> (B, T, N)
    all_acts = jnp.transpose(all_acts, (1, 0, 2))

    # Read output units
    out_idx = jnp.array(topology.output_units)
    logits = all_acts[:, :, out_idx]  # (B, T, output_size)
    return logits


def zero_neg_normalize(logits, eps=1e-10):
    """Lan et al. (2022) output normalization: zero negatives, normalize.

    If all outputs are <= 0, returns uniform 1/n.

    Args:
        logits: (..., n) raw output values.

    Returns:
        probs: (..., n) probability distribution.
    """
    clipped = jnp.maximum(logits, 0.0)
    total = jnp.sum(clipped, axis=-1, keepdims=True)
    n = logits.shape[-1]
    uniform = jnp.ones_like(logits) / n
    # Where total > 0, normalize; otherwise uniform
    probs = jnp.where(total > eps, clipped / (total + eps), uniform)
    return probs


# ---------------------------------------------------------------------------
# Gumbel-Softmax free-form RNN (for differentiable MDL training)
# ---------------------------------------------------------------------------

def _codelength_informed_init(grid_codelengths, scale=1.0):
    """Initialize logits biased toward simple (low-codelength) rationals."""
    cl = jnp.asarray(grid_codelengths)

    def init_fn(rng, shape, dtype=jnp.float32):
        n_total, M = shape
        base = -scale * cl
        noise = jrandom.normal(rng, shape=(n_total, M)) * 0.1
        return jnp.broadcast_to(base[None, :], (n_total, M)) + noise

    return init_fn


class GumbelSoftmaxFreeFormRNN(nn.Module):
    """Free-form RNN with categorical weight parameterization.

    Each connection weight and bias is a categorical distribution over a
    rational grid, trained via Gumbel-Softmax straight-through.

    Attributes:
        topology: FreeFormTopology defining the graph structure.
        grid_values: (M,) rational grid values.
        grid_codelengths: (M,) per-grid-point codelengths.
        mode_forward: use mode of pi instead of Gumbel argmax in forward.
        init_cl_scale: scale for codelength-informed init (0 = noise only).
    """
    topology: FreeFormTopology
    grid_values: Any
    grid_codelengths: Any
    mode_forward: bool = False
    init_cl_scale: float = 0.0

    @nn.compact
    def __call__(self, x, tau, train=True, rng=None,
                 deterministic_st=False):
        """Forward pass through the categorical free-form RNN.

        Args:
            x: int32 (batch, seq_len) input token indices.
            tau: Gumbel-Softmax temperature.
            train: training mode flag.
            rng: PRNG key for Gumbel noise.
            deterministic_st: if True, deterministic straight-through.

        Returns:
            logits: (batch, seq_len, output_size) raw output logits.
            aux: dict with expected_codelength, all_probs, etc.
        """
        topo = self.topology
        n_total = topo.n_weights
        M = len(self.grid_values)

        if self.init_cl_scale > 0:
            init_fn = _codelength_informed_init(
                self.grid_codelengths, scale=self.init_cl_scale,
            )
        else:
            init_fn = nn.initializers.normal(stddev=0.1)
        all_logits = self.param("logits", init_fn, (n_total, M))

        grid = jnp.asarray(self.grid_values)

        # --- Sample / select discrete weights ---
        if train and deterministic_st:
            y_soft = jax.nn.softmax(all_logits / tau, axis=-1)
            idx = jnp.argmax(y_soft, axis=-1)
            y_hard = jax.nn.one_hot(idx, M)
            y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
            all_weights = jnp.sum(y_st * grid[None, :], axis=-1)
        elif train and rng is not None:
            keys = jrandom.split(rng, n_total)
            mode_fwd = jnp.bool_(self.mode_forward)

            def sample_one(logit_row, key):
                gumbel_noise = jrandom.gumbel(key, shape=(M,))
                perturbed = (logit_row + gumbel_noise) / tau
                y_soft = jax.nn.softmax(perturbed, axis=-1)
                idx = jax.lax.cond(
                    mode_fwd,
                    lambda: jnp.argmax(logit_row, axis=-1),
                    lambda: jnp.argmax(y_soft, axis=-1),
                )
                y_hard = jax.nn.one_hot(idx, M)
                y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
                return jnp.dot(y_st, grid)

            all_weights = jax.vmap(sample_one)(all_logits, keys)
        else:
            idx = jnp.argmax(all_logits, axis=-1)
            all_weights = grid[idx]

        # Expected codelength
        all_probs = jax.nn.softmax(all_logits, axis=-1)
        cl = jnp.asarray(self.grid_codelengths)
        expected_codelength = jnp.sum(all_probs * cl[None, :])

        # --- Forward pass with sampled weights ---
        logits = freeform_forward(topo, all_weights, x)

        aux = {
            "expected_codelength": expected_codelength,
            "all_probs": all_probs,
            "all_logits": all_logits,
            "n_params": n_total,
        }
        return logits, aux


def decode_freeform_weights(params, grid_values):
    """Extract discrete weights (argmax) from trained logits."""
    logits = params["params"]["logits"]
    grid = jnp.asarray(grid_values)
    idx = jnp.argmax(logits, axis=-1)
    return grid[idx]
