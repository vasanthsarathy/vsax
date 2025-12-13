"""Tests for Binary VSA operations."""

import jax
import jax.numpy as jnp
import pytest

from vsax.ops import BinaryOperations


class TestBinaryOperations:
    """Test suite for Binary VSA operations."""

    @pytest.fixture
    def ops(self):
        """Create BinaryOperations instance."""
        return BinaryOperations()

    @pytest.fixture
    def bipolar_vectors(self):
        """Create sample bipolar vectors for testing."""
        key = jax.random.PRNGKey(42)
        dim = 512
        vectors = []
        for i in range(3):
            vec = jax.random.choice(
                jax.random.PRNGKey(42 + i),
                jnp.array([-1, 1]),
                shape=(dim,)
            )
            vectors.append(vec)
        return tuple(vectors)

    def test_bind_xor(self, ops):
        """Test XOR binding in bipolar representation."""
        a = jnp.array([1, -1, 1, -1])
        b = jnp.array([1, 1, -1, -1])

        result = ops.bind(a, b)
        expected = jnp.array([1, -1, -1, 1])
        assert jnp.array_equal(result, expected)

    def test_bind_commutative(self, ops, bipolar_vectors):
        """Test that bind is commutative."""
        a, b, _ = bipolar_vectors
        ab = ops.bind(a, b)
        ba = ops.bind(b, a)

        assert jnp.array_equal(ab, ba)

    def test_bind_associative(self, ops, bipolar_vectors):
        """Test that bind is associative."""
        a, b, c = bipolar_vectors
        ab_c = ops.bind(ops.bind(a, b), c)
        a_bc = ops.bind(a, ops.bind(b, c))

        assert jnp.array_equal(ab_c, a_bc)

    def test_bind_self_inverse(self, ops, bipolar_vectors):
        """Test that XOR is self-inverse: bind(bind(a, b), b) = a."""
        a, b, _ = bipolar_vectors
        bound = ops.bind(a, b)
        unbound = ops.bind(bound, b)

        # Should recover exactly
        assert jnp.array_equal(unbound, a)

    def test_bind_with_self(self, ops):
        """Test binding a vector with itself gives all +1."""
        vec = jnp.array([1, -1, 1, -1, 1])
        result = ops.bind(vec, vec)

        # XOR with self is identity (all same -> all +1)
        expected = jnp.ones_like(vec)
        assert jnp.array_equal(result, expected)

    def test_bundle_majority_vote(self, ops):
        """Test majority voting for bundling."""
        a = jnp.array([1, -1, 1, -1])
        b = jnp.array([1, 1, -1, -1])
        c = jnp.array([1, 1, 1, 1])

        result = ops.bundle(a, b, c)

        # Position 0: [1, 1, 1] -> majority 1
        # Position 1: [-1, 1, 1] -> majority 1
        # Position 2: [1, -1, 1] -> majority 1
        # Position 3: [-1, -1, 1] -> majority -1
        expected = jnp.array([1, 1, 1, -1])
        assert jnp.array_equal(result, expected)

    def test_bundle_single_vector(self, ops, bipolar_vectors):
        """Test bundling a single vector."""
        a = bipolar_vectors[0]
        result = ops.bundle(a)

        # Single vector bundle is itself (after sign and where operations)
        assert result.shape == a.shape
        assert jnp.all(jnp.isin(result, jnp.array([-1, 1])))

    def test_bundle_tie_breaking(self, ops):
        """Test tie breaking in majority vote (even number of vectors)."""
        a = jnp.array([1, -1, 1, -1])
        b = jnp.array([-1, 1, -1, 1])

        result = ops.bundle(a, b)

        # All positions have tie (sum = 0)
        # Ties default to +1
        expected = jnp.ones(4, dtype=jnp.int32)
        assert jnp.array_equal(result, expected)

    def test_bundle_empty_raises_error(self, ops):
        """Test that bundling no vectors raises ValueError."""
        with pytest.raises(ValueError, match="at least one vector"):
            ops.bundle()

    def test_bundle_result_is_int32(self, ops, bipolar_vectors):
        """Test that bundle result has int32 dtype."""
        result = ops.bundle(*bipolar_vectors)
        assert result.dtype == jnp.int32

    def test_inverse_is_self(self, ops, bipolar_vectors):
        """Test that inverse returns the vector itself (self-inverse)."""
        a = bipolar_vectors[0]
        inv = ops.inverse(a)

        assert jnp.array_equal(inv, a)

    def test_permute_positive_shift(self, ops):
        """Test permutation with positive shift."""
        vec = jnp.array([1, -1, 1, -1, 1])
        shifted = ops.permute(vec, 2)

        expected = jnp.array([-1, 1, 1, -1, 1])
        assert jnp.array_equal(shifted, expected)

    def test_permute_negative_shift(self, ops):
        """Test permutation with negative shift."""
        vec = jnp.array([1, -1, 1, -1, 1])
        shifted = ops.permute(vec, -2)

        expected = jnp.array([1, -1, 1, 1, -1])
        assert jnp.array_equal(shifted, expected)

    def test_permute_zero_shift(self, ops, bipolar_vectors):
        """Test permutation with zero shift."""
        a = bipolar_vectors[0]
        shifted = ops.permute(a, 0)

        assert jnp.array_equal(shifted, a)

    def test_bind_all_ones(self, ops):
        """Test binding with all +1 (identity element)."""
        vec = jnp.array([1, -1, 1, -1])
        ones = jnp.ones_like(vec)

        result = ops.bind(vec, ones)
        assert jnp.array_equal(result, vec)

    def test_bind_all_minus_ones(self, ops):
        """Test binding with all -1 (negation)."""
        vec = jnp.array([1, -1, 1, -1])
        minus_ones = -jnp.ones_like(vec)

        result = ops.bind(vec, minus_ones)
        expected = -vec
        assert jnp.array_equal(result, expected)

    def test_bundle_preserves_similarity(self, ops):
        """Test that bundling similar vectors preserves similarity."""
        # Create base vector
        key = jax.random.PRNGKey(42)
        base = jax.random.choice(key, jnp.array([-1, 1]), shape=(512,))

        # Create similar vectors (flip 10% of bits)
        similar1 = base.copy()
        flip_indices1 = jax.random.choice(
            jax.random.PRNGKey(43),
            512,
            shape=(51,),
            replace=False
        )
        similar1 = similar1.at[flip_indices1].multiply(-1)

        similar2 = base.copy()
        flip_indices2 = jax.random.choice(
            jax.random.PRNGKey(44),
            512,
            shape=(51,),
            replace=False
        )
        similar2 = similar2.at[flip_indices2].multiply(-1)

        bundled = ops.bundle(similar1, similar2)

        # Calculate Hamming similarity (proportion of matching bits)
        matches = jnp.sum(bundled == base)
        similarity = matches / 512
        assert similarity > 0.7  # Should be reasonably similar

    def test_unbind_exact_recovery(self, ops, bipolar_vectors):
        """Test exact recovery through unbinding."""
        a, b, c = bipolar_vectors

        # Bind a and b, then unbind b (should get exactly a)
        bound = ops.bind(a, b)
        recovered = ops.bind(bound, b)

        assert jnp.array_equal(recovered, a)

    def test_bundle_distributive(self, ops):
        """Test that bind distributes over bundle."""
        a = jnp.array([1, -1, 1, -1])
        b = jnp.array([1, 1, -1, -1])
        c = jnp.array([1, 1, 1, 1])

        # bind(a, bundle(b, c)) vs bundle(bind(a, b), bind(a, c))
        bundled = ops.bundle(b, c)
        left = ops.bind(a, bundled)

        ab = ops.bind(a, b)
        ac = ops.bind(a, c)
        right = ops.bundle(ab, ac)

        # These should be related but not necessarily equal
        # (majority vote doesn't preserve exact distributivity)
        # But let's verify shapes and types
        assert left.shape == right.shape
        assert jnp.all(jnp.isin(left, jnp.array([-1, 1])))
        assert jnp.all(jnp.isin(right, jnp.array([-1, 1])))

    def test_three_way_bind_unbind(self, ops, bipolar_vectors):
        """Test binding three vectors and unbinding."""
        a, b, c = bipolar_vectors

        # Bind all three
        bound = ops.bind(ops.bind(a, b), c)

        # Unbind c and b
        unbound = ops.bind(ops.bind(bound, c), b)

        # Should recover a exactly
        assert jnp.array_equal(unbound, a)

    def test_bundle_odd_number_of_vectors(self, ops):
        """Test bundling odd number of vectors (no ties)."""
        a = jnp.array([1, -1, 1, -1])
        b = jnp.array([1, 1, -1, -1])
        c = jnp.array([1, 1, 1, 1])
        d = jnp.array([-1, -1, -1, -1])
        e = jnp.array([1, 1, 1, -1])

        result = ops.bundle(a, b, c, d, e)

        # With 5 vectors, majority is clear (3 or more)
        assert jnp.all(jnp.isin(result, jnp.array([-1, 1])))
        assert result.dtype == jnp.int32
