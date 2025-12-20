"""Tests for CliffordOperator."""

import jax
import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import CliffordOperator, OperatorKind
from vsax.representations import ComplexHypervector, RealHypervector
from vsax.similarity import cosine_similarity


def test_clifford_operator_creation() -> None:
    """Test basic CliffordOperator creation."""
    dim = 128
    params = jnp.ones(dim) * jnp.pi / 4
    op = CliffordOperator(params=params)

    assert op.dim == dim
    assert op.params.shape == (dim,)
    assert isinstance(op.params, jnp.ndarray)


def test_clifford_operator_creation_from_list() -> None:
    """Test CliffordOperator creation from Python list."""
    params = [0.1, 0.2, 0.3, 0.4]
    op = CliffordOperator(params=params)

    assert op.dim == 4
    assert isinstance(op.params, jnp.ndarray)
    assert jnp.allclose(op.params, jnp.array(params))


def test_clifford_operator_invalid_shape() -> None:
    """Test that multi-dimensional params raises error."""
    params = jnp.ones((10, 10))  # 2D array

    with pytest.raises(ValueError, match="must be 1-dimensional"):
        CliffordOperator(params=params)


def test_clifford_operator_apply() -> None:
    """Test applying operator to hypervector."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    hv = memory["test"]

    result = op.apply(hv)

    assert isinstance(result, ComplexHypervector)
    assert result.vec.shape == (dim,)
    assert jnp.iscomplexobj(result.vec)


def test_clifford_operator_inverse() -> None:
    """Test operator inversion with high precision."""
    dim = 256
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    hv = memory["test"]

    # Apply operator then inverse
    transformed = op.apply(hv)
    recovered = op.inverse().apply(transformed)

    # Should recover original vector with very high similarity
    similarity = cosine_similarity(recovered.vec, hv.vec)
    assert similarity > 0.999, f"Inversion similarity too low: {similarity}"


def test_clifford_operator_inverse_params() -> None:
    """Test that inverse has negated parameters."""
    dim = 128
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])[:dim]
    op = CliffordOperator(params=params)

    inv_op = op.inverse()

    assert jnp.allclose(inv_op.params, -params)


def test_clifford_operator_compose() -> None:
    """Test operator composition."""
    dim = 128
    op1 = CliffordOperator.random(dim, name="op1", key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, name="op2", key=jax.random.PRNGKey(1))

    composed = op1.compose(op2)

    assert composed.dim == dim
    assert jnp.allclose(composed.params, op1.params + op2.params)


def test_clifford_composition_associativity() -> None:
    """Test that composition is associative."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op1 = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, key=jax.random.PRNGKey(1))
    op3 = CliffordOperator.random(dim, key=jax.random.PRNGKey(2))

    hv = memory["test"]

    # (op1 ∘ op2) ∘ op3
    left = op1.compose(op2).compose(op3)
    result_left = left.apply(hv)

    # op1 ∘ (op2 ∘ op3)
    right = op1.compose(op2.compose(op3))
    result_right = right.apply(hv)

    similarity = cosine_similarity(result_left.vec, result_right.vec)
    assert similarity > 0.999, f"Associativity failed: similarity={similarity}"


def test_clifford_composition_commutativity() -> None:
    """Test that composition is commutative (for phase addition)."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op1 = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, key=jax.random.PRNGKey(1))

    hv = memory["test"]

    # op1 ∘ op2
    comp_12 = op1.compose(op2)
    result_12 = comp_12.apply(hv)

    # op2 ∘ op1
    comp_21 = op2.compose(op1)
    result_21 = comp_21.apply(hv)

    similarity = cosine_similarity(result_12.vec, result_21.vec)
    assert similarity > 0.999, f"Commutativity failed: similarity={similarity}"


def test_clifford_composition_inverse() -> None:
    """Test inverse of composed operator."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op1 = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, key=jax.random.PRNGKey(1))

    hv = memory["test"]

    # Compose and invert
    composed = op1.compose(op2)
    composed_inv = composed.inverse()

    # Apply composed then inverse
    transformed = composed.apply(hv)
    recovered = composed_inv.apply(transformed)

    similarity = cosine_similarity(recovered.vec, hv.vec)
    assert similarity > 0.999


def test_clifford_dimension_mismatch_compose() -> None:
    """Test that dimension mismatch in compose raises error."""
    op1 = CliffordOperator.random(128, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(256, key=jax.random.PRNGKey(1))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        op1.compose(op2)


def test_clifford_dimension_mismatch_apply() -> None:
    """Test that dimension mismatch in apply raises error."""
    dim = 128
    model = create_fhrr_model(dim=256)  # Different dimension
    memory = VSAMemory(model)
    memory.add("test")

    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        op.apply(memory["test"])


def test_clifford_requires_complex_hypervector() -> None:
    """Test that operator requires ComplexHypervector."""
    dim = 128
    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))

    # Try with RealHypervector (should fail)
    real_hv = RealHypervector(jnp.ones(dim))

    with pytest.raises(TypeError, match="ComplexHypervector"):
        op.apply(real_hv)


def test_clifford_compose_wrong_type() -> None:
    """Test that composing with non-CliffordOperator raises error."""
    op = CliffordOperator.random(128, key=jax.random.PRNGKey(0))

    with pytest.raises(TypeError, match="Can only compose with CliffordOperator"):
        op.compose("not an operator")  # type: ignore


def test_clifford_metadata() -> None:
    """Test operator metadata handling."""
    dim = 128
    op = CliffordOperator.random(
        dim, kind=OperatorKind.SPATIAL, name="TEST_OP", key=jax.random.PRNGKey(0)
    )

    assert op.metadata is not None
    assert op.metadata.kind == OperatorKind.SPATIAL
    assert op.metadata.name == "TEST_OP"
    assert op.metadata.invertible is True


def test_clifford_random_reproducible() -> None:
    """Test that random operator is reproducible with same key."""
    dim = 128
    key = jax.random.PRNGKey(42)

    op1 = CliffordOperator.random(dim, key=key)
    op2 = CliffordOperator.random(dim, key=key)

    assert jnp.allclose(op1.params, op2.params)


def test_clifford_random_different_keys() -> None:
    """Test that different keys produce different operators."""
    dim = 128

    op1 = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, key=jax.random.PRNGKey(1))

    assert not jnp.allclose(op1.params, op2.params)


def test_clifford_repr() -> None:
    """Test string representation of operator."""
    op = CliffordOperator.random(
        128, kind=OperatorKind.SPATIAL, name="LEFT_OF", key=jax.random.PRNGKey(0)
    )

    repr_str = repr(op)
    assert "LEFT_OF" in repr_str
    assert "dim=128" in repr_str
    assert "spatial" in repr_str


def test_clifford_compose_metadata() -> None:
    """Test that composed operator has correct metadata."""
    op1 = CliffordOperator.random(128, name="OP1", key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(128, name="OP2", key=jax.random.PRNGKey(1))

    composed = op1.compose(op2)

    assert composed.metadata is not None
    assert "compose" in composed.metadata.name
    assert "OP1" in composed.metadata.name
    assert "OP2" in composed.metadata.name


def test_clifford_immutability() -> None:
    """Test that CliffordOperator is immutable."""
    op = CliffordOperator.random(128, key=jax.random.PRNGKey(0))

    # Should not be able to modify params
    with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
        op.params = jnp.ones(128)  # type: ignore


def test_clifford_phase_range() -> None:
    """Test that random operator has phases in [0, 2π)."""
    dim = 256
    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))

    # All phases should be in [0, 2π)
    assert jnp.all(op.params >= 0)
    assert jnp.all(op.params < 2 * jnp.pi)


def test_clifford_preserves_magnitude() -> None:
    """Test that operator preserves unit magnitude of FHRR vectors."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    hv = memory["test"]

    # FHRR vectors have unit magnitude
    original_mag = jnp.abs(hv.vec)
    assert jnp.allclose(original_mag, 1.0, atol=1e-5)

    # After transformation, should still have unit magnitude
    transformed = op.apply(hv)
    transformed_mag = jnp.abs(transformed.vec)
    assert jnp.allclose(transformed_mag, 1.0, atol=1e-5)


def test_clifford_sequential_operations() -> None:
    """Test multiple sequential applications and inversions."""
    dim = 128
    model = create_fhrr_model(dim=dim)
    memory = VSAMemory(model)
    memory.add("test")

    op1 = CliffordOperator.random(dim, key=jax.random.PRNGKey(0))
    op2 = CliffordOperator.random(dim, key=jax.random.PRNGKey(1))
    op3 = CliffordOperator.random(dim, key=jax.random.PRNGKey(2))

    hv = memory["test"]

    # Apply sequence: op1 -> op2 -> op3
    result = op1.apply(hv)
    result = op2.apply(result)
    result = op3.apply(result)

    # Undo in reverse: op3^-1 -> op2^-1 -> op1^-1
    result = op3.inverse().apply(result)
    result = op2.inverse().apply(result)
    result = op1.inverse().apply(result)

    # Should recover original
    similarity = cosine_similarity(result.vec, hv.vec)
    assert similarity > 0.999
