"""Tests for quaternion algebra functions."""

import jax.numpy as jnp

from vsax.ops.quaternion import (
    qconj,
    qidentity,
    qinverse,
    qmul,
    qnorm,
    qnorm_squared,
    qnormalize,
)


class TestQuaternionAlgebra:
    """Test suite for pure quaternion algebra functions."""

    # =====================================================================
    # Hamilton Product (qmul) Tests
    # =====================================================================

    def test_qmul_identity(self):
        """Test multiplication with identity quaternion."""
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        q = jnp.array([1.0, 2.0, 3.0, 4.0])

        # identity * q = q
        result = qmul(identity, q)
        assert jnp.allclose(result, q)

        # q * identity = q
        result = qmul(q, identity)
        assert jnp.allclose(result, q)

    def test_qmul_basis_quaternions(self):
        """Test Hamilton product of basis quaternions i, j, k."""
        # Basis quaternions: 1, i, j, k
        one = jnp.array([1.0, 0.0, 0.0, 0.0])
        i = jnp.array([0.0, 1.0, 0.0, 0.0])
        j = jnp.array([0.0, 0.0, 1.0, 0.0])
        k = jnp.array([0.0, 0.0, 0.0, 1.0])

        # i² = j² = k² = -1
        assert jnp.allclose(qmul(i, i), -one)
        assert jnp.allclose(qmul(j, j), -one)
        assert jnp.allclose(qmul(k, k), -one)

        # i * j = k
        assert jnp.allclose(qmul(i, j), k)

        # j * k = i
        assert jnp.allclose(qmul(j, k), i)

        # k * i = j
        assert jnp.allclose(qmul(k, i), j)

        # j * i = -k (anti-commutativity)
        assert jnp.allclose(qmul(j, i), -k)

        # k * j = -i
        assert jnp.allclose(qmul(k, j), -i)

        # i * k = -j
        assert jnp.allclose(qmul(i, k), -j)

    def test_qmul_non_commutative(self):
        """Test that Hamilton product is non-commutative."""
        p = jnp.array([1.0, 2.0, 3.0, 4.0])
        q = jnp.array([5.0, 6.0, 7.0, 8.0])

        pq = qmul(p, q)
        qp = qmul(q, p)

        # pq != qp in general
        assert not jnp.allclose(pq, qp)

    def test_qmul_associative(self):
        """Test that Hamilton product is associative."""
        p = jnp.array([1.0, 2.0, 3.0, 4.0])
        q = jnp.array([5.0, 6.0, 7.0, 8.0])
        r = jnp.array([9.0, 10.0, 11.0, 12.0])

        # (p * q) * r = p * (q * r)
        pq_r = qmul(qmul(p, q), r)
        p_qr = qmul(p, qmul(q, r))

        assert jnp.allclose(pq_r, p_qr, atol=1e-6)

    def test_qmul_batched(self):
        """Test Hamilton product with batched inputs."""
        p = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        q = jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])  # identities

        result = qmul(p, q)

        assert result.shape == (2, 4)
        assert jnp.allclose(result, p)

    def test_qmul_inverse_identity(self):
        """Test that q * q⁻¹ = identity."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        q_inv = qinverse(q)
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])

        result = qmul(q, q_inv)
        assert jnp.allclose(result, identity, atol=1e-6)

    # =====================================================================
    # Conjugate (qconj) Tests
    # =====================================================================

    def test_qconj_basic(self):
        """Test quaternion conjugate."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        expected = jnp.array([1.0, -2.0, -3.0, -4.0])

        result = qconj(q)
        assert jnp.allclose(result, expected)

    def test_qconj_double_conjugate(self):
        """Test that double conjugate returns original."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = qconj(qconj(q))
        assert jnp.allclose(result, q)

    def test_qconj_batched(self):
        """Test conjugate with batched inputs."""
        q = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        expected = jnp.array([[1.0, -2.0, -3.0, -4.0], [5.0, -6.0, -7.0, -8.0]])

        result = qconj(q)
        assert jnp.allclose(result, expected)

    # =====================================================================
    # Norm Tests
    # =====================================================================

    def test_qnorm_squared(self):
        """Test squared norm calculation."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        expected = 1.0 + 4.0 + 9.0 + 16.0  # = 30

        result = qnorm_squared(q)
        assert jnp.allclose(result, expected)

    def test_qnorm(self):
        """Test norm calculation."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        expected = jnp.sqrt(30.0)

        result = qnorm(q)
        assert jnp.allclose(result, expected)

    def test_qnorm_unit_quaternion(self):
        """Test norm of unit quaternion is 1."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])  # Already unit

        result = qnorm(q)
        assert jnp.allclose(result, 1.0)

    # =====================================================================
    # Inverse (qinverse) Tests
    # =====================================================================

    def test_qinverse_identity(self):
        """Test inverse of identity is identity."""
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])

        result = qinverse(identity)
        assert jnp.allclose(result, identity)

    def test_qinverse_unit_quaternion(self):
        """Test inverse of unit quaternion equals conjugate."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])  # Unit quaternion

        inv = qinverse(q)
        conj = qconj(q)

        assert jnp.allclose(inv, conj, atol=1e-6)

    def test_qinverse_general(self):
        """Test inverse of general quaternion."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])

        q_inv = qinverse(q)

        # q * q⁻¹ = identity
        result = qmul(q, q_inv)
        assert jnp.allclose(result, identity, atol=1e-6)

    def test_qinverse_batched(self):
        """Test inverse with batched inputs."""
        q = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])

        q_inv = qinverse(q)

        # Each should satisfy q * q⁻¹ = identity
        for i in range(2):
            result = qmul(q[i], q_inv[i])
            assert jnp.allclose(result, identity, atol=1e-6)

    # =====================================================================
    # Normalize (qnormalize) Tests
    # =====================================================================

    def test_qnormalize_produces_unit(self):
        """Test that normalize produces unit quaternions."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = qnormalize(q)
        norm = qnorm(result)

        assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_qnormalize_preserves_direction(self):
        """Test that normalize preserves direction."""
        q = jnp.array([2.0, 0.0, 0.0, 0.0])

        result = qnormalize(q)
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])

        assert jnp.allclose(result, expected)

    def test_qnormalize_batched(self):
        """Test normalize with batched inputs."""
        q = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        result = qnormalize(q)

        # All should have unit norm
        norms = qnorm(result)
        assert jnp.allclose(norms, jnp.ones(2), atol=1e-6)

    # =====================================================================
    # Identity (qidentity) Tests
    # =====================================================================

    def test_qidentity_single(self):
        """Test single identity quaternion."""
        result = qidentity()
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])

        assert result.shape == (4,)
        assert jnp.allclose(result, expected)

    def test_qidentity_batched(self):
        """Test batched identity quaternions."""
        result = qidentity((10,))

        assert result.shape == (10, 4)
        assert jnp.all(result[:, 0] == 1.0)
        assert jnp.all(result[:, 1:] == 0.0)

    def test_qidentity_multidim(self):
        """Test multi-dimensional identity quaternions."""
        result = qidentity((3, 5))

        assert result.shape == (3, 5, 4)
        assert jnp.all(result[..., 0] == 1.0)
        assert jnp.all(result[..., 1:] == 0.0)

    # =====================================================================
    # Property Tests
    # =====================================================================

    def test_norm_product_rule(self):
        """Test |p * q| = |p| * |q|."""
        p = jnp.array([1.0, 2.0, 3.0, 4.0])
        q = jnp.array([5.0, 6.0, 7.0, 8.0])

        product = qmul(p, q)
        norm_product = qnorm(product)
        norm_p_times_norm_q = qnorm(p) * qnorm(q)

        assert jnp.allclose(norm_product, norm_p_times_norm_q, atol=1e-5)

    def test_conjugate_product_rule(self):
        """Test (p * q)* = q* * p*."""
        p = jnp.array([1.0, 2.0, 3.0, 4.0])
        q = jnp.array([5.0, 6.0, 7.0, 8.0])

        # (p * q)*
        conj_product = qconj(qmul(p, q))

        # q* * p*
        product_conj = qmul(qconj(q), qconj(p))

        assert jnp.allclose(conj_product, product_conj, atol=1e-6)

    def test_unit_quaternion_multiplication_closed(self):
        """Test that product of unit quaternions is unit."""
        p = qnormalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        q = qnormalize(jnp.array([5.0, 6.0, 7.0, 8.0]))

        product = qmul(p, q)
        norm = qnorm(product)

        assert jnp.allclose(norm, 1.0, atol=1e-6)
