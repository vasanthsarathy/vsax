"""Tests for spatial operators."""

import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import (
    OperatorKind,
    create_above,
    create_behind,
    create_below,
    create_far,
    create_in_front_of,
    create_left_of,
    create_near,
    create_right_of,
    create_spatial_operators,
)
from vsax.similarity import cosine_similarity


def test_create_left_of() -> None:
    """Test creating LEFT_OF operator."""
    op = create_left_of(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "LEFT_OF"
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_create_right_of() -> None:
    """Test creating RIGHT_OF operator."""
    op = create_right_of(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "LEFT_OF"  # RIGHT_OF is inverse of LEFT_OF
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_left_right_inverse_pair() -> None:
    """Test that LEFT_OF and RIGHT_OF are exact inverses."""
    LEFT_OF = create_left_of(512)
    RIGHT_OF = create_right_of(512)

    # RIGHT_OF should be inverse of LEFT_OF
    assert jnp.allclose(RIGHT_OF.params, -LEFT_OF.params)

    # Composing them should give near-identity (params sum to zero)
    composed = LEFT_OF.compose(RIGHT_OF)
    # The composed operator should have parameters close to zero
    # (which means it's close to identity transformation)
    assert jnp.allclose(composed.params, jnp.zeros(512), atol=1e-5)


def test_create_above() -> None:
    """Test creating ABOVE operator."""
    op = create_above(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "ABOVE"
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_create_below() -> None:
    """Test creating BELOW operator."""
    op = create_below(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "ABOVE"  # BELOW is inverse of ABOVE
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_above_below_inverse_pair() -> None:
    """Test that ABOVE and BELOW are exact inverses."""
    ABOVE = create_above(512)
    BELOW = create_below(512)

    # BELOW should be inverse of ABOVE
    assert jnp.allclose(BELOW.params, -ABOVE.params)

    # Composing them should give near-identity
    composed = ABOVE.compose(BELOW)
    assert jnp.allclose(composed.params, jnp.zeros(512), atol=1e-5)


def test_create_in_front_of() -> None:
    """Test creating IN_FRONT_OF operator."""
    op = create_in_front_of(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "IN_FRONT_OF"
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_create_behind() -> None:
    """Test creating BEHIND operator."""
    op = create_behind(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "IN_FRONT_OF"  # BEHIND is inverse of IN_FRONT_OF
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_in_front_behind_inverse_pair() -> None:
    """Test that IN_FRONT_OF and BEHIND are exact inverses."""
    IN_FRONT_OF = create_in_front_of(512)
    BEHIND = create_behind(512)

    # BEHIND should be inverse of IN_FRONT_OF
    assert jnp.allclose(BEHIND.params, -IN_FRONT_OF.params)


def test_create_near() -> None:
    """Test creating NEAR operator."""
    op = create_near(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "NEAR"
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_create_far() -> None:
    """Test creating FAR operator."""
    op = create_far(512)

    assert op.dim == 512
    assert op.metadata is not None
    assert op.metadata.name == "NEAR"  # FAR is inverse of NEAR
    assert op.metadata.kind == OperatorKind.SPATIAL


def test_near_far_inverse_pair() -> None:
    """Test that NEAR and FAR are exact inverses."""
    NEAR = create_near(512)
    FAR = create_far(512)

    # FAR should be inverse of NEAR
    assert jnp.allclose(FAR.params, -NEAR.params)


def test_spatial_operators_reproducible() -> None:
    """Test that spatial operators are reproducible."""
    # Create operators twice
    LEFT_OF_1 = create_left_of(512)
    LEFT_OF_2 = create_left_of(512)

    ABOVE_1 = create_above(512)
    ABOVE_2 = create_above(512)

    # Should have identical parameters
    assert jnp.allclose(LEFT_OF_1.params, LEFT_OF_2.params)
    assert jnp.allclose(ABOVE_1.params, ABOVE_2.params)


def test_spatial_operators_different() -> None:
    """Test that different spatial operators have different parameters."""
    LEFT_OF = create_left_of(512)
    ABOVE = create_above(512)
    NEAR = create_near(512)

    # Different operators should have different parameters
    assert not jnp.allclose(LEFT_OF.params, ABOVE.params)
    assert not jnp.allclose(LEFT_OF.params, NEAR.params)
    assert not jnp.allclose(ABOVE.params, NEAR.params)


def test_create_spatial_operators() -> None:
    """Test creating all spatial operators at once."""
    operators = create_spatial_operators(512)

    # Should have all 8 operators
    assert len(operators) == 8
    assert "LEFT_OF" in operators
    assert "RIGHT_OF" in operators
    assert "ABOVE" in operators
    assert "BELOW" in operators
    assert "IN_FRONT_OF" in operators
    assert "BEHIND" in operators
    assert "NEAR" in operators
    assert "FAR" in operators

    # All should have correct dimension
    for op in operators.values():
        assert op.dim == 512
        assert op.metadata.kind == OperatorKind.SPATIAL


def test_spatial_operators_integration() -> None:
    """Test spatial operators with actual hypervectors."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["cup", "plate", "table"])

    LEFT_OF = create_left_of(512)

    # Encode spatial relation: cup, LEFT_OF(plate)
    # Bundle them together into a scene
    scene = model.opset.bundle(memory["cup"].vec, LEFT_OF.apply(memory["plate"]).vec)

    # Query: what's LEFT_OF plate? (apply inverse operator to scene)
    answer = LEFT_OF.inverse().apply(model.rep_cls(scene))

    # Should have high similarity to plate (which got transformed)
    similarity_cup = cosine_similarity(answer.vec, memory["cup"].vec)
    similarity_plate = cosine_similarity(answer.vec, memory["plate"].vec)
    similarity_table = cosine_similarity(answer.vec, memory["table"].vec)

    # Plate should have highest similarity (it's the one that was transformed)
    assert similarity_plate > similarity_cup
    assert similarity_plate > similarity_table
    assert similarity_plate > 0.4  # Should have meaningful similarity


def test_spatial_operators_complex_scene() -> None:
    """Test encoding complex spatial scene with multiple relations."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["cup", "plate", "table"])

    LEFT_OF = create_left_of(512)
    ABOVE = create_above(512)

    # Encode: cup, LEFT_OF(plate), ABOVE(table)
    scene = model.opset.bundle(
        memory["cup"].vec, LEFT_OF.apply(memory["plate"]).vec, ABOVE.apply(memory["table"]).vec
    )

    # Query 1: what's LEFT_OF transformed?
    answer1 = LEFT_OF.inverse().apply(model.rep_cls(scene))
    sim_plate = cosine_similarity(answer1.vec, memory["plate"].vec)

    # Query 2: what's ABOVE transformed?
    answer2 = ABOVE.inverse().apply(model.rep_cls(scene))
    sim_table = cosine_similarity(answer2.vec, memory["table"].vec)

    # Both queries should retrieve the transformed items
    assert sim_plate > 0.3
    assert sim_table > 0.3


def test_spatial_operators_composition() -> None:
    """Test composing spatial operators."""
    LEFT_OF = create_left_of(512)
    ABOVE = create_above(512)

    # Compose: move left then up
    left_and_up = LEFT_OF.compose(ABOVE)

    assert left_and_up.dim == 512
    # Composed params should be sum of individual params
    assert jnp.allclose(left_and_up.params, LEFT_OF.params + ABOVE.params)


def test_spatial_operators_exact_inversion() -> None:
    """Test exact inversion with spatial operators."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    LEFT_OF = create_left_of(512)

    hv = memory["test"]
    transformed = LEFT_OF.apply(hv)
    recovered = LEFT_OF.inverse().apply(transformed)

    # Should recover original with high similarity
    similarity = cosine_similarity(recovered.vec, hv.vec)
    assert similarity > 0.999


def test_spatial_operators_different_dimensions() -> None:
    """Test creating spatial operators with different dimensions."""
    op_512 = create_left_of(512)
    op_1024 = create_left_of(1024)

    assert op_512.dim == 512
    assert op_1024.dim == 1024

    # Different dimensions should have different parameter vectors
    assert op_512.params.shape[0] == 512
    assert op_1024.params.shape[0] == 1024
