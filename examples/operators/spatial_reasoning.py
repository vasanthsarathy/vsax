"""Example: Spatial Reasoning with Clifford Operators.

Demonstrates how to use spatial operators to encode and query spatial layouts.

This example shows:
- Creating spatial operators (LEFT_OF, ABOVE, etc.)
- Encoding spatial scenes with multiple objects
- Querying spatial relations
- Composing spatial transformations
"""

import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import create_above, create_left_of, create_near
from vsax.similarity import cosine_similarity


def main() -> None:
    """Run spatial reasoning example."""
    print("=" * 80)
    print("Spatial Reasoning with Clifford Operators")
    print("=" * 80)
    print()

    # Setup
    print("Setting up FHRR model with dimension 1024...")
    model = create_fhrr_model(dim=1024)
    memory = VSAMemory(model)

    # Add objects
    objects = ["cup", "plate", "table", "lamp", "book"]
    memory.add_many(objects)
    print(f"Added {len(objects)} objects to memory")
    print()

    # Create spatial operators
    print("Creating spatial operators...")
    LEFT_OF = create_left_of(1024)
    RIGHT_OF = LEFT_OF.inverse()
    ABOVE = create_above(1024)
    BELOW = ABOVE.inverse()
    NEAR = create_near(1024)

    print(f"  {LEFT_OF}")
    print(f"  {RIGHT_OF}")
    print(f"  {ABOVE}")
    print(f"  {BELOW}")
    print(f"  {NEAR}")
    print()

    # Example 1: Simple spatial relation
    print("-" * 80)
    print("Example 1: Simple Spatial Relation")
    print("-" * 80)
    print()

    print("Encoding: 'cup is LEFT_OF plate'")
    scene1 = model.opset.bundle(
        memory["cup"].vec, LEFT_OF.apply(memory["plate"]).vec
    )

    print("Query: What has the LEFT_OF transformation applied?")
    answer1 = LEFT_OF.inverse().apply(model.rep_cls(scene1))

    for obj in objects:
        sim = cosine_similarity(answer1.vec, memory[obj].vec)
        print(f"  Similarity to '{obj}': {sim:.3f}")

    print("\n-> 'plate' has highest similarity (it was transformed by LEFT_OF)")
    print()

    # Example 2: Complex spatial scene
    print("-" * 80)
    print("Example 2: Complex Spatial Scene")
    print("-" * 80)
    print()

    print("Encoding spatial layout:")
    print("  - cup LEFT_OF plate")
    print("  - plate ABOVE table")
    print("  - lamp NEAR book")

    scene2 = model.opset.bundle(
        memory["cup"].vec,
        LEFT_OF.apply(memory["plate"]).vec,
        memory["plate"].vec,
        ABOVE.apply(memory["table"]).vec,
        memory["lamp"].vec,
        NEAR.apply(memory["book"]).vec,
    )

    print("\nQuery 1: What has LEFT_OF applied?")
    answer2a = LEFT_OF.inverse().apply(model.rep_cls(scene2))
    sim_plate = cosine_similarity(answer2a.vec, memory["plate"].vec)
    print(f"  Similarity to 'plate': {sim_plate:.3f}")

    print("\nQuery 2: What has ABOVE applied?")
    answer2b = ABOVE.inverse().apply(model.rep_cls(scene2))
    sim_table = cosine_similarity(answer2b.vec, memory["table"].vec)
    print(f"  Similarity to 'table': {sim_table:.3f}")

    print("\nQuery 3: What has NEAR applied?")
    answer2c = NEAR.inverse().apply(model.rep_cls(scene2))
    sim_book = cosine_similarity(answer2c.vec, memory["book"].vec)
    print(f"  Similarity to 'book': {sim_book:.3f}")

    print(
        "\n-> All queries successfully retrieve the transformed objects with high similarity"
    )
    print()

    # Example 3: Operator composition
    print("-" * 80)
    print("Example 3: Composing Spatial Transformations")
    print("-" * 80)
    print()

    print("Creating composed operator: LEFT_OF + ABOVE (move left and up)")
    left_and_up = LEFT_OF.compose(ABOVE)
    print(f"  Composed operator: {left_and_up}")

    print("\nVerifying composition is equivalent to sequential application:")
    test_vec = memory["cup"]

    # Method 1: Apply composed operator
    result1 = left_and_up.apply(test_vec)

    # Method 2: Apply operators sequentially
    result2 = ABOVE.apply(LEFT_OF.apply(test_vec))

    # They should be identical
    similarity = cosine_similarity(result1.vec, result2.vec)
    print(f"  Similarity between composed and sequential: {similarity:.6f}")
    print("  -> Composition works correctly (similarity ~= 1.0)")
    print()

    # Example 4: Exact inversion
    print("-" * 80)
    print("Example 4: Exact Inversion Property")
    print("-" * 80)
    print()

    print("Testing exact inversion with spatial operators...")
    original = memory["cup"]

    # Apply LEFT_OF transformation
    transformed = LEFT_OF.apply(original)
    print("  Applied LEFT_OF to 'cup'")

    # Apply inverse (RIGHT_OF)
    recovered = RIGHT_OF.apply(transformed)
    print("  Applied RIGHT_OF (inverse) to result")

    # Check recovery accuracy
    recovery_sim = cosine_similarity(recovered.vec, original.vec)
    print(f"\n  Recovery similarity: {recovery_sim:.6f}")
    print(
        f"  -> Exact inversion: {recovery_sim > 0.999} (similarity > 0.999)"
    )
    print()

    # Example 5: Multiple inverse pairs
    print("-" * 80)
    print("Example 5: Testing All Inverse Pairs")
    print("-" * 80)
    print()

    print("Verifying inverse pair properties...")
    test_obj = memory["book"]

    # Test LEFT_OF / RIGHT_OF
    recovered1 = RIGHT_OF.apply(LEFT_OF.apply(test_obj))
    sim1 = cosine_similarity(recovered1.vec, test_obj.vec)
    print(f"  LEFT_OF + RIGHT_OF: {sim1:.6f}")

    # Test ABOVE / BELOW
    recovered2 = BELOW.apply(ABOVE.apply(test_obj))
    sim2 = cosine_similarity(recovered2.vec, test_obj.vec)
    print(f"  ABOVE + BELOW: {sim2:.6f}")

    # Verify parameters
    print("\nVerifying parameter relationships:")
    print(
        f"  RIGHT_OF params = -LEFT_OF params: {jnp.allclose(RIGHT_OF.params, -LEFT_OF.params)}"
    )
    print(
        f"  BELOW params = -ABOVE params: {jnp.allclose(BELOW.params, -ABOVE.params)}"
    )
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("* Spatial operators encode directional relations")
    print("* Inverse operators enable exact querying")
    print("* Operators can be composed for complex transformations")
    print("* Inversion is exact (similarity > 0.999)")
    print("* All inverse pairs work correctly")
    print()
    print("Use cases:")
    print("  - Robot navigation and spatial reasoning")
    print("  - Scene understanding and object localization")
    print("  - Spatial question answering")
    print("  - Geographic information systems")
    print()


if __name__ == "__main__":
    main()
