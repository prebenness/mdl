"""Tests for the free-form RNN module and coding scheme."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fractions import Fraction

from src.mdl.freeform_rnn import (
    FreeFormTopology,
    freeform_forward,
    zero_neg_normalize,
    GumbelSoftmaxFreeFormRNN,
    ACTIVATION_BITS,
)
from src.mdl.freeform_coding import freeform_codelength
from src.mdl.coding import integer_code_length, rational_codelength


# ---------------------------------------------------------------------------
# Topology construction helpers
# ---------------------------------------------------------------------------

class TestFreeFormTopology:

    def test_basic_construction(self):
        topo = FreeFormTopology(
            n_units=3,
            activations=("linear", "relu", "linear"),
            connections=(("input", 0, 0), ("forward", 0, 1)),
            biased_units=frozenset(),
            output_units=(2,),
            input_size=2,
            output_size=1,
        )
        assert topo.n_units == 3
        assert topo.n_connection_weights == 2
        assert topo.n_bias_weights == 0
        assert topo.n_weights == 2

    def test_forward_order_violation_raises(self):
        with pytest.raises(AssertionError, match="topological"):
            FreeFormTopology(
                n_units=3,
                activations=("linear", "linear", "linear"),
                connections=(("forward", 2, 1),),  # 2 > 1!
                biased_units=frozenset(),
                output_units=(2,),
                input_size=2,
                output_size=1,
            )

    def test_recurrent_self_loop_ok(self):
        """Recurrent connections don't need topological ordering."""
        topo = FreeFormTopology(
            n_units=2,
            activations=("linear", "linear"),
            connections=(("recurrent", 1, 1),),  # self-loop OK
            biased_units=frozenset(),
            output_units=(1,),
            input_size=1,
            output_size=1,
        )
        assert topo.n_weights == 1

    def test_biased_units(self):
        topo = FreeFormTopology(
            n_units=3,
            activations=("linear", "linear", "linear"),
            connections=(),
            biased_units=frozenset({0, 2}),
            output_units=(1,),
            input_size=1,
            output_size=1,
        )
        assert topo.n_bias_weights == 2
        assert topo.sorted_biased_units() == [0, 2]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestFreeFormForward:

    def test_linear_passthrough(self):
        """A single linear unit passing input through."""
        topo = FreeFormTopology(
            n_units=1,
            activations=("linear",),
            connections=(("input", 0, 0),),
            biased_units=frozenset(),
            output_units=(0,),
            input_size=2,
            output_size=1,
        )
        weights = jnp.array([1.0])  # weight on input connection
        x = jnp.array([[0, 1, 0]], dtype=jnp.int32)  # tokens: 0, 1, 0
        logits = freeform_forward(topo, weights, x)
        # Token 0 -> one_hot = [1,0], input dim 0 -> 1.0
        # Token 1 -> one_hot = [0,1], input dim 0 -> 0.0
        assert logits.shape == (1, 3, 1)
        np.testing.assert_allclose(logits[0, :, 0], [1.0, 0.0, 1.0])

    def test_relu_activation(self):
        topo = FreeFormTopology(
            n_units=1,
            activations=("relu",),
            connections=(("input", 0, 0),),
            biased_units=frozenset({0}),
            output_units=(0,),
            input_size=2,
            output_size=1,
        )
        # weight=-1 on input, bias=0.5 -> pre = -1*x[0] + 0.5
        weights = jnp.array([-1.0, 0.5])  # connection, then bias
        x = jnp.array([[0, 1]], dtype=jnp.int32)
        logits = freeform_forward(topo, weights, x)
        # Token 0: pre = -1*1 + 0.5 = -0.5, relu = 0
        # Token 1: pre = -1*0 + 0.5 = 0.5, relu = 0.5
        np.testing.assert_allclose(logits[0, :, 0], [0.0, 0.5], atol=1e-6)

    def test_recurrent_counter(self):
        """A unit that counts via recurrent self-loop."""
        topo = FreeFormTopology(
            n_units=1,
            activations=("linear",),
            connections=(
                ("input", 0, 0),     # input dim 0 -> unit 0
                ("recurrent", 0, 0),  # self-loop
            ),
            biased_units=frozenset(),
            output_units=(0,),
            input_size=2,
            output_size=1,
        )
        # w_input=1, w_recurrent=1 -> accumulates input[0]
        weights = jnp.array([1.0, 1.0])
        # Tokens: 0, 0, 1, 0 -> one_hot dim 0: 1, 1, 0, 1
        x = jnp.array([[0, 0, 1, 0]], dtype=jnp.int32)
        logits = freeform_forward(topo, weights, x)
        # t=0: 1*1 + 1*0 = 1
        # t=1: 1*1 + 1*1 = 2
        # t=2: 1*0 + 1*2 = 2
        # t=3: 1*1 + 1*2 = 3
        np.testing.assert_allclose(logits[0, :, 0], [1, 2, 2, 3], atol=1e-5)

    def test_forward_connection_order(self):
        """Forward connections respect topological order."""
        topo = FreeFormTopology(
            n_units=2,
            activations=("linear", "linear"),
            connections=(
                ("input", 0, 0),      # input -> unit 0
                ("forward", 0, 1),    # unit 0 -> unit 1 (same timestep)
            ),
            biased_units=frozenset(),
            output_units=(1,),
            input_size=2,
            output_size=1,
        )
        weights = jnp.array([2.0, 3.0])  # input*2 -> unit0, unit0*3 -> unit1
        x = jnp.array([[0, 1]], dtype=jnp.int32)
        logits = freeform_forward(topo, weights, x)
        # Token 0: unit0 = 2*1 = 2, unit1 = 3*2 = 6
        # Token 1: unit0 = 2*0 = 0, unit1 = 3*0 = 0
        np.testing.assert_allclose(logits[0, :, 0], [6.0, 0.0], atol=1e-5)

    def test_batch_dimension(self):
        topo = FreeFormTopology(
            n_units=1,
            activations=("linear",),
            connections=(("input", 0, 0),),
            biased_units=frozenset(),
            output_units=(0,),
            input_size=2,
            output_size=1,
        )
        weights = jnp.array([1.0])
        x = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)  # batch of 2
        logits = freeform_forward(topo, weights, x)
        assert logits.shape == (2, 2, 1)
        np.testing.assert_allclose(logits[0, :, 0], [1.0, 0.0])
        np.testing.assert_allclose(logits[1, :, 0], [0.0, 1.0])


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

class TestZeroNegNormalize:

    def test_positive_values(self):
        logits = jnp.array([2.0, 1.0, 0.0])
        probs = zero_neg_normalize(logits)
        np.testing.assert_allclose(probs, [2/3, 1/3, 0.0], atol=1e-6)

    def test_negative_zeroed(self):
        logits = jnp.array([3.0, -1.0, 0.0])
        probs = zero_neg_normalize(logits)
        np.testing.assert_allclose(probs, [1.0, 0.0, 0.0], atol=1e-6)

    def test_all_negative_uniform(self):
        logits = jnp.array([-1.0, -2.0, -3.0])
        probs = zero_neg_normalize(logits)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)

    def test_batched(self):
        logits = jnp.array([[3.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        probs = zero_neg_normalize(logits)
        np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(probs[1], [1/3, 1/3, 1/3], atol=1e-6)


# ---------------------------------------------------------------------------
# Coding scheme
# ---------------------------------------------------------------------------

class TestFreeFormCoding:

    def test_integer_code_sanity(self):
        """Verify our integer_code_length matches Lan et al. examples."""
        assert integer_code_length(0) == 1
        assert integer_code_length(1) == 3
        assert integer_code_length(2) == 5
        assert integer_code_length(5) == 7

    def test_minimal_network_codelength(self):
        """A trivial 1-unit network with no connections."""
        topo = FreeFormTopology(
            n_units=1,
            activations=("linear",),
            connections=(),
            biased_units=frozenset(),
            output_units=(0,),
            input_size=1,
            output_size=1,
        )
        result = freeform_codelength(topo, [], {})
        # E(1 unit) = 3 bits
        # Unit 0: linear(0) + E(0 outgoing)(1) + no bias = 1 bit
        # Total = 3 + 1 = 4
        assert result["total_bits"] == 4

    def test_single_connection_codelength(self):
        """One unit with one outgoing connection, weight = 1/1."""
        topo = FreeFormTopology(
            n_units=2,
            activations=("linear", "linear"),
            connections=(("forward", 0, 1),),
            biased_units=frozenset(),
            output_units=(1,),
            input_size=1,
            output_size=1,
        )
        w = Fraction(1, 1)
        result = freeform_codelength(topo, [w], {})

        # E(2 units) = 5 bits
        # Unit 0: linear(0) + E(1 outgoing)(3) + [E(1) + 1 + rational(1/1)]
        #   E(target=1) = 3, type = 1, weight(1/1) = 1 + E(1) + E(1) = 1+3+3 = 7
        #   connection total = 3 + 1 + 7 = 11
        #   Unit 0 total = 0 + 3 + 11 = 14
        # Unit 1: linear(0) + E(0 outgoing)(1) = 1
        # Total = 5 + 14 + 1 = 20
        assert result["total_bits"] == 20

    def test_bias_encoding(self):
        """Unit with a non-zero bias adds to codelength."""
        topo = FreeFormTopology(
            n_units=1,
            activations=("linear",),
            connections=(),
            biased_units=frozenset({0}),
            output_units=(0,),
            input_size=1,
            output_size=1,
        )
        bias = Fraction(1, 2)
        result = freeform_codelength(topo, [], {0: bias})
        # E(1) = 3
        # Unit 0: linear(0) + E(0)(1) + rational(1/2)
        #   rational(1/2) = 1 + E(1) + E(2) = 1 + 3 + 5 = 9
        # Total = 3 + 0 + 1 + 9 = 13
        assert result["total_bits"] == 13

    def test_activation_cost(self):
        """Non-linear activations add encoding cost."""
        for act, expected_cost in ACTIVATION_BITS.items():
            topo = FreeFormTopology(
                n_units=1,
                activations=(act,),
                connections=(),
                biased_units=frozenset(),
                output_units=(0,),
                input_size=1,
                output_size=1,
            )
            result = freeform_codelength(topo, [], {})
            # E(1) = 3, unit: act_cost + E(0) = act_cost + 1
            assert result["total_bits"] == 3 + expected_cost + 1, (
                f"activation {act}: expected {3 + expected_cost + 1}, "
                f"got {result['total_bits']}"
            )


# ---------------------------------------------------------------------------
# GumbelSoftmax module
# ---------------------------------------------------------------------------

class TestGumbelSoftmaxFreeFormRNN:

    def _make_simple_topo(self):
        return FreeFormTopology(
            n_units=2,
            activations=("linear", "linear"),
            connections=(
                ("input", 0, 0),
                ("forward", 0, 1),
            ),
            biased_units=frozenset(),
            output_units=(1,),
            input_size=3,
            output_size=1,
        )

    def test_init_and_forward(self):
        topo = self._make_simple_topo()
        grid_values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        grid_cl = np.array([7.0, 5.0, 7.0], dtype=np.float32)

        model = GumbelSoftmaxFreeFormRNN(
            topology=topo,
            grid_values=grid_values,
            grid_codelengths=grid_cl,
        )
        rng = jax.random.PRNGKey(0)
        x = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        params = model.init(rng, x, tau=1.0, train=False)
        logits, aux = model.apply(params, x, tau=1.0, train=False)
        assert logits.shape == (1, 3, 1)
        assert "expected_codelength" in aux
        assert aux["n_params"] == 2

    def test_deterministic_st(self):
        topo = self._make_simple_topo()
        grid_values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        grid_cl = np.array([7.0, 5.0, 7.0], dtype=np.float32)

        model = GumbelSoftmaxFreeFormRNN(
            topology=topo,
            grid_values=grid_values,
            grid_codelengths=grid_cl,
        )
        rng = jax.random.PRNGKey(42)
        x = jnp.array([[0, 1]], dtype=jnp.int32)
        params = model.init(rng, x, tau=1.0, train=False)
        logits, aux = model.apply(
            params, x, tau=0.5, train=True, deterministic_st=True,
        )
        assert logits.shape == (1, 2, 1)
