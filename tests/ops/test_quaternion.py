"""Tests for QuaternionOperations."""

import jax
import jax.numpy as jnp
import pytest

from vsax.ops.quaternion import (
    QuaternionOperations,
    qidentity,
    qinverse,
    qmul,
    qnormalize,
    sandwich,
    sandwich_unit,
)


class TestQuaternionOperations:
    """Test suite for QuaternionOperations."""

    @pytest.fixture
    def ops(self):
        """Create QuaternionOperations instance."""
        return QuaternionOperations()

    @pytest.fixture
    def quaternion_vectors(self):
        """Create sample quaternion hypervectors for testing."""
        key = jax.random.PRNGKey(42)
        dim = 256
        n = 3

        # Sample random quaternions and normalize
        raw = jax.random.normal(key, shape=(n, dim, 4))
        norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / (norms + 1e-10)

    # =====================================================================
    # Bind Tests
    # =====================================================================

    def test_bind_shape(self, ops, quaternion_vectors):
        """Test that bind preserves shape."""
        a, b, _ = quaternion_vectors

        result = ops.bind(a, b)

        assert result.shape == a.shape

    def test_bind_non_commutative(self, ops, quaternion_vectors):
        """Test that bind is NON-commutative."""
        a, b, _ = quaternion_vectors

        ab = ops.bind(a, b)
        ba = ops.bind(b, a)

        # ab != ba for quaternions
        assert not jnp.allclose(ab, ba)

    def test_bind_non_commutative_similarity(self, ops, quaternion_vectors):
        """Test that bind(x,y) and bind(y,x) have low similarity."""
        a, b, _ = quaternion_vectors

        ab = ops.bind(a, b)
        ba = ops.bind(b, a)

        # Compute similarity (average dot product of quaternions)
        similarity = jnp.mean(jnp.sum(ab * ba, axis=-1))

        # Should be low (not highly similar)
        assert similarity < 0.5, f"Non-commutative similarity {similarity} too high"

    def test_bind_associative(self, ops, quaternion_vectors):
        """Test that bind is associative."""
        a, b, c = quaternion_vectors

        ab_c = ops.bind(ops.bind(a, b), c)
        a_bc = ops.bind(a, ops.bind(b, c))

        assert jnp.allclose(ab_c, a_bc, atol=1e-5)

    def test_bind_with_identity(self, ops, quaternion_vectors):
        """Test binding with identity quaternion."""
        a, _, _ = quaternion_vectors
        dim = a.shape[0]
        identity = qidentity((dim,))

        # a * identity = a
        result = ops.bind(a, identity)
        assert jnp.allclose(result, a, atol=1e-6)

        # identity * a = a
        result = ops.bind(identity, a)
        assert jnp.allclose(result, a, atol=1e-6)

    # =====================================================================
    # Bundle Tests
    # =====================================================================

    def test_bundle_single_vector(self, ops, quaternion_vectors):
        """Test bundling a single vector."""
        a = quaternion_vectors[0]

        result = ops.bundle(a)

        # Should normalize to unit quaternions
        norms = jnp.linalg.norm(result, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_bundle_multiple_vectors(self, ops, quaternion_vectors):
        """Test bundling multiple vectors."""
        result = ops.bundle(*quaternion_vectors)

        assert result.shape == quaternion_vectors[0].shape

        # Should be normalized
        norms = jnp.linalg.norm(result, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_bundle_empty_raises_error(self, ops):
        """Test that bundling no vectors raises ValueError."""
        with pytest.raises(ValueError, match="at least one vector"):
            ops.bundle()

    def test_bundle_preserves_similarity(self, ops, quaternion_vectors):
        """Test that bundled vector is similar to constituents."""
        a, b, c = quaternion_vectors

        bundled = ops.bundle(a, b, c)

        # Compute similarity with each input
        sim_a = jnp.mean(jnp.sum(bundled * a, axis=-1))
        sim_b = jnp.mean(jnp.sum(bundled * b, axis=-1))
        sim_c = jnp.mean(jnp.sum(bundled * c, axis=-1))

        # All should have positive similarity
        assert sim_a > 0
        assert sim_b > 0
        assert sim_c > 0

    # =====================================================================
    # Inverse Tests
    # =====================================================================

    def test_inverse_identity(self, ops, quaternion_vectors):
        """Test that a * a⁻¹ = identity."""
        a, _, _ = quaternion_vectors
        identity = qidentity((a.shape[0],))

        a_inv = ops.inverse(a)
        result = ops.bind(a, a_inv)

        assert jnp.allclose(result, identity, atol=1e-5)

    def test_inverse_of_inverse(self, ops, quaternion_vectors):
        """Test that (a⁻¹)⁻¹ = a."""
        a, _, _ = quaternion_vectors

        double_inv = ops.inverse(ops.inverse(a))

        assert jnp.allclose(double_inv, a, atol=1e-5)

    # =====================================================================
    # Right-Unbind Tests
    # =====================================================================

    def test_unbind_right_recovers_x(self, ops, quaternion_vectors):
        """Test that right-unbind recovers x from bind(x, y)."""
        x, y, _ = quaternion_vectors

        # z = x * y
        z = ops.bind(x, y)

        # Right-unbind: z * y⁻¹ = x
        recovered_x = ops.unbind(z, y)

        assert jnp.allclose(recovered_x, x, atol=1e-5)

    def test_unbind_right_high_similarity(self, ops, quaternion_vectors):
        """Test that right-unbind achieves high similarity."""
        x, y, _ = quaternion_vectors

        z = ops.bind(x, y)
        recovered_x = ops.unbind(z, y)

        # Compute similarity
        similarity = jnp.mean(jnp.sum(x * recovered_x, axis=-1))

        assert similarity > 0.99, f"Right-unbind similarity {similarity} too low"

    # =====================================================================
    # Left-Unbind Tests
    # =====================================================================

    def test_unbind_left_recovers_y(self, ops, quaternion_vectors):
        """Test that left-unbind recovers y from bind(x, y)."""
        x, y, _ = quaternion_vectors

        # z = x * y
        z = ops.bind(x, y)

        # Left-unbind: x⁻¹ * z = y
        recovered_y = ops.unbind_left(x, z)

        assert jnp.allclose(recovered_y, y, atol=1e-5)

    def test_unbind_left_high_similarity(self, ops, quaternion_vectors):
        """Test that left-unbind achieves high similarity."""
        x, y, _ = quaternion_vectors

        z = ops.bind(x, y)
        recovered_y = ops.unbind_left(x, z)

        # Compute similarity
        similarity = jnp.mean(jnp.sum(y * recovered_y, axis=-1))

        assert similarity > 0.99, f"Left-unbind similarity {similarity} too low"

    def test_unbind_left_different_from_right(self, ops, quaternion_vectors):
        """Test that left-unbind gives different result than right-unbind."""
        x, y, _ = quaternion_vectors

        z = ops.bind(x, y)

        # Right-unbind recovers x
        right_result = ops.unbind(z, y)

        # Left-unbind recovers y
        left_result = ops.unbind_left(x, z)

        # Should be different (x != y)
        assert not jnp.allclose(right_result, left_result)

    # =====================================================================
    # Round-Trip Tests
    # =====================================================================

    def test_bind_unbind_chain(self, ops, quaternion_vectors):
        """Test bind-unbind chain: bind(a,b,c) -> unbind c -> unbind b -> a."""
        a, b, c = quaternion_vectors

        # Bind all three
        bound = ops.bind(ops.bind(a, b), c)

        # Unbind step by step (right-unbind)
        step1 = ops.unbind(bound, c)
        step2 = ops.unbind(step1, b)

        # Should recover a
        assert jnp.allclose(step2, a, atol=1e-4)

    def test_left_right_unbind_chain(self, ops, quaternion_vectors):
        """Test chained unbinding: (x * y) with both left and right unbind."""
        x, y, _ = quaternion_vectors

        z = ops.bind(x, y)  # z = x * y

        # Right-unbind to get x
        x_recovered = ops.unbind(z, y)  # z * y⁻¹

        # Left-unbind to get y
        y_recovered = ops.unbind_left(x, z)  # x⁻¹ * z

        assert jnp.allclose(x_recovered, x, atol=1e-5)
        assert jnp.allclose(y_recovered, y, atol=1e-5)

    # =====================================================================
    # Permute Tests
    # =====================================================================

    def test_permute_shifts_correctly(self, ops):
        """Test that permute shifts along quaternion coordinate dimension."""
        # Simple test case
        vec = jnp.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]])

        shifted = ops.permute(vec, 1)

        expected = jnp.array([[4, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])
        assert jnp.array_equal(shifted, expected)

    def test_permute_zero_shift(self, ops, quaternion_vectors):
        """Test that zero shift returns original."""
        a, _, _ = quaternion_vectors

        shifted = ops.permute(a, 0)

        assert jnp.array_equal(shifted, a)

    def test_permute_full_cycle(self, ops, quaternion_vectors):
        """Test that shifting by length returns original."""
        a, _, _ = quaternion_vectors
        dim = a.shape[-2]

        shifted = ops.permute(a, dim)

        assert jnp.allclose(shifted, a)

    # =====================================================================
    # Property Tests
    # =====================================================================

    def test_unit_quaternions_preserved_under_bind(self, ops, quaternion_vectors):
        """Test that binding unit quaternions produces unit quaternions."""
        a, b, _ = quaternion_vectors

        result = ops.bind(a, b)
        norms = jnp.linalg.norm(result, axis=-1)

        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_order_sensitivity(self, ops, quaternion_vectors):
        """Test that binding is order-sensitive (role vs filler matters)."""
        role, filler, _ = quaternion_vectors

        # role * filler
        role_filler = ops.bind(role, filler)

        # filler * role
        filler_role = ops.bind(filler, role)

        # These should be different
        diff = jnp.mean(jnp.abs(role_filler - filler_role))
        assert diff > 0.1, f"Expected order sensitivity, but diff={diff}"

        # Unbinding should work correctly
        recovered_role = ops.unbind(role_filler, filler)
        recovered_filler = ops.unbind_left(role, role_filler)

        assert jnp.allclose(recovered_role, role, atol=1e-5)
        assert jnp.allclose(recovered_filler, filler, atol=1e-5)


# =============================================================================
# Sandwich Product Tests
# =============================================================================


class TestSandwichProduct:
    """Test suite for sandwich product operations."""

    @pytest.fixture
    def quaternion_vectors(self):
        """Create sample quaternion hypervectors for testing."""
        key = jax.random.PRNGKey(42)
        dim = 256
        n = 3

        # Sample random quaternions and normalize
        raw = jax.random.normal(key, shape=(n, dim, 4))
        norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / (norms + 1e-10)

    @pytest.fixture
    def ops(self):
        """Create QuaternionOperations instance."""
        return QuaternionOperations()

    def test_sandwich_identity(self, quaternion_vectors):
        """Test that sandwich(identity, v) == v."""
        v, _, _ = quaternion_vectors
        dim = v.shape[0]
        identity = qidentity((dim,))

        result = sandwich(identity, v)

        assert jnp.allclose(result, v, atol=1e-5)

    def test_sandwich_unit_identity(self, quaternion_vectors):
        """Test that sandwich_unit(identity, v) == v."""
        v, _, _ = quaternion_vectors
        dim = v.shape[0]
        identity = qidentity((dim,))

        result = sandwich_unit(identity, v)

        assert jnp.allclose(result, v, atol=1e-5)

    def test_sandwich_inverse_roundtrip(self, quaternion_vectors):
        """Test that sandwich(q, sandwich(q^-1, v)) recovers v."""
        rotor, v, _ = quaternion_vectors
        rotor_inv = qinverse(rotor)

        # Transform v with rotor inverse, then with rotor
        transformed = sandwich(rotor_inv, v)
        recovered = sandwich(rotor, transformed)

        assert jnp.allclose(recovered, v, atol=1e-4)

    def test_sandwich_unit_equivalent(self, quaternion_vectors):
        """Test that sandwich_unit equals sandwich for unit quaternions."""
        rotor, v, _ = quaternion_vectors

        # Both should give same result for unit quaternions
        result1 = sandwich(rotor, v)
        result2 = sandwich_unit(rotor, v)

        assert jnp.allclose(result1, result2, atol=1e-5)

    def test_sandwich_composition(self, quaternion_vectors):
        """Test that sandwich(q2, sandwich(q1, v)) == sandwich(q2*q1, v)."""
        q1, q2, v = quaternion_vectors

        # Sequential application
        step1 = sandwich(q1, v)
        sequential = sandwich(q2, step1)

        # Composed rotor
        composed_rotor = qmul(q2, q1)
        direct = sandwich(composed_rotor, v)

        assert jnp.allclose(sequential, direct, atol=1e-4)

    def test_sandwich_preserves_unit_length(self, quaternion_vectors):
        """Test that sandwich preserves unit quaternion length."""
        rotor, v, _ = quaternion_vectors

        result = sandwich(rotor, v)
        norms = jnp.linalg.norm(result, axis=-1)

        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_sandwich_ops_method(self, ops, quaternion_vectors):
        """Test that ops.sandwich matches standalone sandwich function."""
        rotor, v, _ = quaternion_vectors

        result1 = sandwich(rotor, v)
        result2 = ops.sandwich(rotor, v)

        assert jnp.allclose(result1, result2)

    def test_sandwich_unit_ops_method(self, ops, quaternion_vectors):
        """Test that ops.sandwich_unit matches standalone function."""
        rotor, v, _ = quaternion_vectors

        result1 = sandwich_unit(rotor, v)
        result2 = ops.sandwich_unit(rotor, v)

        assert jnp.allclose(result1, result2)

    def test_sandwich_different_from_bind(self, quaternion_vectors):
        """Test that sandwich gives different result than simple binding."""
        rotor, v, _ = quaternion_vectors

        sandwich_result = sandwich(rotor, v)
        bind_result = qmul(rotor, v)

        # These should NOT be equal
        assert not jnp.allclose(sandwich_result, bind_result)

    def test_sandwich_single_quaternion(self):
        """Test sandwich product on single quaternions (not hypervectors)."""
        # 90-degree rotation around x-axis: cos(45) + sin(45)i
        rotor = qnormalize(jnp.array([1.0, 1.0, 0.0, 0.0]))
        v = jnp.array([0.0, 0.0, 1.0, 0.0])  # Pure j quaternion

        result = sandwich_unit(rotor, v)

        # Result should be a pure quaternion (real part near 0)
        assert jnp.abs(result[0]) < 0.1
