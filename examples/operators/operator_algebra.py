"""Example: Operator Algebra and Properties.

Demonstrates the algebraic properties of Clifford operators.

This example shows:
- Exact inversion property
- Associative composition
- Commutative composition (for phase-based operators)
- Operator reproducibility
- Performance characteristics
"""

import time

import jax
import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import CliffordOperator, OperatorKind, create_agent, create_left_of
from vsax.similarity import cosine_similarity


def main() -> None:
    """Run operator algebra example."""
    print("=" * 80)
    print("Operator Algebra and Properties")
    print("=" * 80)
    print()

    # Setup
    print("Setting up FHRR model with dimension 1024...")
    model = create_fhrr_model(dim=1024)
    memory = VSAMemory(model)
    memory.add_many(["test1", "test2", "test3"])
    print()

    # Property 1: Exact Inversion
    print("-" * 80)
    print("Property 1: Exact Inversion")
    print("-" * 80)
    print()

    print("Creating operator and testing inversion...")
    op = CliffordOperator.random(1024, key=jax.random.PRNGKey(0))

    test_vec = memory["test1"]
    transformed = op.apply(test_vec)
    recovered = op.inverse().apply(transformed)

    similarity = cosine_similarity(recovered.vec, test_vec.vec)
    print(f"  Recovery similarity: {similarity:.10f}")
    print(f"  Exact inversion (>0.999): {similarity > 0.999} *")

    # Test with multiple vectors
    print("\nTesting with multiple random vectors:")
    for i in range(5):
        key = jax.random.PRNGKey(i)
        test_op = CliffordOperator.random(1024, key=key)
        test_hv = memory["test1"]

        result = test_op.inverse().apply(test_op.apply(test_hv))
        sim = cosine_similarity(result.vec, test_hv.vec)
        print(f"  Vector {i + 1}: {sim:.10f} (exact: {sim > 0.999})")
    print()

    # Property 2: Associativity
    print("-" * 80)
    print("Property 2: Associative Composition")
    print("-" * 80)
    print()

    print("Testing: (op1 o op2) o op3 = op1 o (op2 o op3)")

    op1 = CliffordOperator.random(1024, name="OP1", key=jax.random.PRNGKey(1))
    op2 = CliffordOperator.random(1024, name="OP2", key=jax.random.PRNGKey(2))
    op3 = CliffordOperator.random(1024, name="OP3", key=jax.random.PRNGKey(3))

    # Left grouping: (op1 o op2) o op3
    left = op1.compose(op2).compose(op3)
    left_result = left.apply(test_vec)

    # Right grouping: op1 o (op2 o op3)
    right = op1.compose(op2.compose(op3))
    right_result = right.apply(test_vec)

    similarity = cosine_similarity(left_result.vec, right_result.vec)
    print(f"  Similarity: {similarity:.10f}")
    print(f"  Associative (>0.999): {similarity > 0.999} *")

    # Verify parameters
    print("\nParameter-level verification:")
    print(
        f"  Left params ~= Right params: {jnp.allclose(left.params, right.params, atol=1e-5)}"
    )
    print()

    # Property 3: Commutativity
    print("-" * 80)
    print("Property 3: Commutative Composition")
    print("-" * 80)
    print()

    print("Testing: op1 o op2 = op2 o op1 (for phase-based operators)")

    # Forward composition
    comp_12 = op1.compose(op2)
    result_12 = comp_12.apply(test_vec)

    # Reverse composition
    comp_21 = op2.compose(op1)
    result_21 = comp_21.apply(test_vec)

    similarity = cosine_similarity(result_12.vec, result_21.vec)
    print(f"  Similarity: {similarity:.10f}")
    print(f"  Commutative (>0.999): {similarity > 0.999} *")

    # Verify parameters
    print("\nParameter-level verification:")
    print(
        f"  Params equal: {jnp.allclose(comp_12.params, comp_21.params, atol=1e-5)}"
    )
    print(
        "  Reason: Phase addition is commutative (params1 + params2 = params2 + params1)"
    )
    print()

    # Property 4: Inverse of Composition
    print("-" * 80)
    print("Property 4: Inverse of Composition")
    print("-" * 80)
    print()

    print("Testing: (op1 o op2)^(-1) o (op1 o op2) = identity")

    composed = op1.compose(op2)
    composed_inv = composed.inverse()

    # Apply composed then inverse
    transformed = composed.apply(test_vec)
    recovered = composed_inv.apply(transformed)

    similarity = cosine_similarity(recovered.vec, test_vec.vec)
    print(f"  Recovery similarity: {similarity:.10f}")
    print(f"  Exact recovery (>0.999): {similarity > 0.999} *")

    print("\nTesting: (op1 o op2)^(-1) = op2^(-1) o op1^(-1)")
    method1 = composed.inverse()
    method2 = op2.inverse().compose(op1.inverse())

    print(
        f"  Parameters equal: {jnp.allclose(method1.params, method2.params, atol=1e-5)}"
    )
    print()

    # Property 5: Reproducibility
    print("-" * 80)
    print("Property 5: Reproducibility")
    print("-" * 80)
    print()

    print("Testing that same seed produces same operator...")

    # Create operator twice with same key
    key = jax.random.PRNGKey(42)
    op_a = CliffordOperator.random(1024, key=key)
    op_b = CliffordOperator.random(1024, key=key)

    print(f"  Parameters identical: {jnp.allclose(op_a.params, op_b.params)}")

    # Test pre-defined operators
    print("\nTesting pre-defined operators...")
    left1 = create_left_of(1024)
    left2 = create_left_of(1024)

    print(f"  create_left_of() params identical: {jnp.allclose(left1.params, left2.params)}")

    agent1 = create_agent(1024)
    agent2 = create_agent(1024)

    print(f"  create_agent() params identical: {jnp.allclose(agent1.params, agent2.params)}")
    print()

    # Property 6: Norm Preservation
    print("-" * 80)
    print("Property 6: Norm Preservation")
    print("-" * 80)
    print()

    print("Testing that operators preserve unit magnitude (for FHRR)...")

    test_hv = memory["test1"]
    print(f"  Original magnitude: {jnp.abs(test_hv.vec).mean():.6f}")

    transformed = op.apply(test_hv)
    print(f"  Transformed magnitude: {jnp.abs(transformed.vec).mean():.6f}")

    magnitude_preserved = jnp.allclose(
        jnp.abs(test_hv.vec), jnp.abs(transformed.vec), atol=1e-5
    )
    print(f"\n  Magnitude preserved: {magnitude_preserved}")
    print("  -> Phase rotation preserves magnitude *")
    print()

    # Property 7: Identity and Inverse
    print("-" * 80)
    print("Property 7: Identity Through Inverse Composition")
    print("-" * 80)
    print()

    print("Testing: op o op^(-1) ~= identity")

    identity = op.compose(op.inverse())
    print(f"  Identity operator params: {identity.params[:5]}...")  # Show first 5
    print(f"  All params near zero: {jnp.allclose(identity.params, 0, atol=1e-5)}")

    # Apply identity to a vector
    identity_result = identity.apply(test_vec)
    similarity = cosine_similarity(identity_result.vec, test_vec.vec)
    print(f"\n  Identity similarity: {similarity:.10f}")
    print(
        f"  Acts as identity (>0.999): {similarity > 0.999} *"
    )
    print()

    # Property 8: Sequential vs Composed
    print("-" * 80)
    print("Property 8: Sequential Application vs Composition")
    print("-" * 80)
    print()

    print("Comparing: compose(op1, op2).apply(v) vs op2.apply(op1.apply(v))")

    # Method 1: Compose then apply
    start = time.time()
    composed_op = op1.compose(op2)
    comp_result = composed_op.apply(test_vec)
    comp_time = time.time() - start

    # Method 2: Apply sequentially
    start = time.time()
    seq_result = op2.apply(op1.apply(test_vec))
    seq_time = time.time() - start

    similarity = cosine_similarity(comp_result.vec, seq_result.vec)
    print(f"  Results identical: {similarity > 0.999} *")
    print(f"\n  Composition time: {comp_time * 1000:.3f} ms")
    print(f"  Sequential time: {seq_time * 1000:.3f} ms")
    print(
        "  -> Composition allows operator precomputation"
    )
    print()

    # Property 9: Operator Metadata
    print("-" * 80)
    print("Property 9: Operator Metadata and Typing")
    print("-" * 80)
    print()

    print("Creating operators with different metadata...")

    spatial_op = CliffordOperator.random(
        1024,
        kind=OperatorKind.SPATIAL,
        name="MY_SPATIAL",
        key=jax.random.PRNGKey(100),
    )

    semantic_op = CliffordOperator.random(
        1024,
        kind=OperatorKind.SEMANTIC,
        name="MY_SEMANTIC",
        key=jax.random.PRNGKey(200),
    )

    print(f"  Spatial operator: {spatial_op}")
    print(f"    Kind: {spatial_op.metadata.kind.value}")
    print(f"    Name: {spatial_op.metadata.name}")
    print(f"    Invertible: {spatial_op.metadata.invertible}")

    print(f"\n  Semantic operator: {semantic_op}")
    print(f"    Kind: {semantic_op.metadata.kind.value}")
    print(f"    Name: {semantic_op.metadata.name}")
    print(f"    Invertible: {semantic_op.metadata.invertible}")

    # Compose operators with metadata
    composed_meta = spatial_op.compose(semantic_op)
    print(f"\n  Composed operator: {composed_meta}")
    print("    Metadata preserved in composition *")
    print()

    # Summary
    print("=" * 80)
    print("Summary: Operator Algebra Properties")
    print("=" * 80)
    print()
    print("Mathematical Properties:")
    print("  * Exact inversion: op^(-1) o op = identity (similarity > 0.999)")
    print("  * Associativity: (op1 o op2) o op3 = op1 o (op2 o op3)")
    print("  * Commutativity: op1 o op2 = op2 o op1 (for phase operators)")
    print("  * Inverse composition: (op1 o op2)^(-1) = op2^(-1) o op1^(-1)")
    print("  * Norm preservation: |op(v)| = |v| (for FHRR)")
    print()
    print("Implementation Properties:")
    print("  * Reproducibility: Same seed -> same operator")
    print("  * Composition optimization: Can precompute composed operators")
    print("  * Metadata tracking: Semantic typing and naming")
    print("  * Type safety: Only works with ComplexHypervector")
    print()
    print("Phase-Based Implementation:")
    print("  - Elementary operators: phase generators (exp(i * params))")
    print("  - Composition: phase addition (params1 + params2)")
    print("  - Inversion: phase negation (-params)")
    print("  - Application: element-wise complex multiplication")
    print()
    print("Use Cases for Operator Algebra:")
    print("  - Building complex transformations from simple ones")
    print("  - Exact query mechanisms with perfect inversion")
    print("  - Compositional reasoning with multiple relations")
    print("  - Typed symbolic manipulation")
    print()


if __name__ == "__main__":
    main()
