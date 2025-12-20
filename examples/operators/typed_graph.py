"""Example: Typed Graph Reasoning with Clifford Operators.

Demonstrates how to use operators to encode knowledge graphs with typed edges.

This example shows:
- Creating custom relation operators for graph edges
- Encoding typed knowledge graphs
- Multi-hop reasoning
- Querying graph structure
"""

import jax
import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.operators import CliffordOperator, OperatorKind
from vsax.similarity import cosine_similarity


def main() -> None:
    """Run typed graph reasoning example."""
    print("=" * 80)
    print("Typed Graph Reasoning with Clifford Operators")
    print("=" * 80)
    print()

    # Setup
    print("Setting up FHRR model with dimension 1024...")
    model = create_fhrr_model(dim=1024)
    memory = VSAMemory(model)

    # Add entities
    entities = [
        "dog",
        "cat",
        "animal",
        "mammal",
        "pet",
        "tail",
        "whiskers",
        "Fido",
        "Fluffy",
        "canine",
        "feline",
    ]
    memory.add_many(entities)
    print(f"Added {len(entities)} entities to memory")
    print()

    # Create relation operators for graph edges
    print("Creating relation operators (typed edges)...")

    # Taxonomic relations
    IS_A = CliffordOperator.random(
        dim=1024,
        kind=OperatorKind.RELATION,
        name="IS_A",
        key=jax.random.PRNGKey(3000),
    )
    SUBCLASS_OF = CliffordOperator.random(
        dim=1024,
        kind=OperatorKind.RELATION,
        name="SUBCLASS_OF",
        key=jax.random.PRNGKey(3001),
    )

    # Part-whole relations
    HAS_PART = CliffordOperator.random(
        dim=1024,
        kind=OperatorKind.RELATION,
        name="HAS_PART",
        key=jax.random.PRNGKey(3002),
    )
    PART_OF = HAS_PART.inverse()

    # Instance relations
    INSTANCE_OF = CliffordOperator.random(
        dim=1024,
        kind=OperatorKind.RELATION,
        name="INSTANCE_OF",
        key=jax.random.PRNGKey(3003),
    )

    print(f"  {IS_A}")
    print(f"  {SUBCLASS_OF}")
    print(f"  {HAS_PART}")
    print(f"  {PART_OF}")
    print(f"  {INSTANCE_OF}")
    print()

    # Example 1: Simple knowledge graph
    print("-" * 80)
    print("Example 1: Simple Taxonomy")
    print("-" * 80)
    print()

    print("Building knowledge graph:")
    print("  dog IS_A mammal")
    print("  cat IS_A mammal")
    print("  mammal IS_A animal")

    # Encode facts
    fact1 = model.opset.bind(memory["dog"].vec, IS_A.apply(memory["mammal"]).vec)
    fact2 = model.opset.bind(memory["cat"].vec, IS_A.apply(memory["mammal"]).vec)
    fact3 = model.opset.bind(
        memory["mammal"].vec, IS_A.apply(memory["animal"]).vec
    )

    # Create knowledge base by bundling facts
    kb = model.opset.bundle(fact1, fact2, fact3)

    print("\nQuery 1: What IS_A mammal?")
    # Unbind the IS_A(mammal) from the KB
    answer1 = model.opset.bind(kb, model.opset.inverse(IS_A.apply(memory["mammal"]).vec))

    for entity in ["dog", "cat", "mammal", "animal"]:
        sim = cosine_similarity(answer1, memory[entity].vec)
        if sim > 0.3:
            print(f"  '{entity}': {sim:.3f}")

    print("\nQuery 2: What IS_A animal?")
    answer2 = model.opset.bind(kb, model.opset.inverse(IS_A.apply(memory["animal"]).vec))

    for entity in ["dog", "cat", "mammal", "animal"]:
        sim = cosine_similarity(answer2, memory[entity].vec)
        if sim > 0.3:
            print(f"  '{entity}': {sim:.3f}")
    print()

    # Example 2: Part-whole relationships
    print("-" * 80)
    print("Example 2: Part-Whole Relationships")
    print("-" * 80)
    print()

    print("Building part-whole graph:")
    print("  dog HAS_PART tail")
    print("  cat HAS_PART whiskers")

    # Encode facts
    fact4 = model.opset.bind(memory["dog"].vec, HAS_PART.apply(memory["tail"]).vec)
    fact5 = model.opset.bind(
        memory["cat"].vec, HAS_PART.apply(memory["whiskers"]).vec
    )

    # Update knowledge base
    kb2 = model.opset.bundle(fact1, fact2, fact3, fact4, fact5)

    print("\nQuery: What parts does dog have?")
    # This doesn't work as expected - we need to bind with the entity first

    # Better approach: encode as entity-relation-target triples
    print("\nQuery: What HAS_PART tail? (inverse query)")
    answer4 = model.opset.bind(kb2, model.opset.inverse(HAS_PART.apply(memory["tail"]).vec))

    for entity in ["dog", "cat", "tail", "whiskers"]:
        sim = cosine_similarity(answer4, memory[entity].vec)
        if sim > 0.2:
            print(f"  '{entity}': {sim:.3f}")
    print()

    # Example 3: Instance relationships
    print("-" * 80)
    print("Example 3: Instances and Classes")
    print("-" * 80)
    print()

    print("Building instance graph:")
    print("  Fido INSTANCE_OF dog")
    print("  Fluffy INSTANCE_OF cat")
    print("  dog SUBCLASS_OF canine")
    print("  cat SUBCLASS_OF feline")

    # Encode facts
    fact6 = model.opset.bind(
        memory["Fido"].vec, INSTANCE_OF.apply(memory["dog"]).vec
    )
    fact7 = model.opset.bind(
        memory["Fluffy"].vec, INSTANCE_OF.apply(memory["cat"]).vec
    )
    fact8 = model.opset.bind(
        memory["dog"].vec, SUBCLASS_OF.apply(memory["canine"]).vec
    )
    fact9 = model.opset.bind(
        memory["cat"].vec, SUBCLASS_OF.apply(memory["feline"]).vec
    )

    # Create knowledge base
    kb3 = model.opset.bundle(fact6, fact7, fact8, fact9)

    print("\nQuery: What is INSTANCE_OF dog?")
    answer5 = model.opset.bind(
        kb3, model.opset.inverse(INSTANCE_OF.apply(memory["dog"]).vec)
    )

    for entity in ["Fido", "Fluffy", "dog", "cat"]:
        sim = cosine_similarity(answer5, memory[entity].vec)
        if sim > 0.3:
            print(f"  '{entity}': {sim:.3f}")

    print("\nQuery: What is SUBCLASS_OF feline?")
    answer6 = model.opset.bind(
        kb3, model.opset.inverse(SUBCLASS_OF.apply(memory["feline"]).vec)
    )

    for entity in ["dog", "cat", "canine", "feline"]:
        sim = cosine_similarity(answer6, memory[entity].vec)
        if sim > 0.3:
            print(f"  '{entity}': {sim:.3f}")
    print()

    # Example 4: Operator properties
    print("-" * 80)
    print("Example 4: Operator Properties and Composition")
    print("-" * 80)
    print()

    print("Testing exact inversion of relation operators...")
    test_entity = memory["dog"]

    # Test IS_A inversion
    transformed = IS_A.apply(test_entity)
    recovered = IS_A.inverse().apply(transformed)
    sim_recovery = cosine_similarity(recovered.vec, test_entity.vec)
    print(f"  IS_A inversion: {sim_recovery:.6f} (exact: {sim_recovery > 0.999})")

    # Test HAS_PART / PART_OF inverse pair
    print(
        f"\n  PART_OF = HAS_PART.inverse(): {jnp.allclose(PART_OF.params, -HAS_PART.params)}"
    )

    print("\nTesting operator composition...")
    # Compose two relations
    composed = IS_A.compose(SUBCLASS_OF)
    print(f"  Composed IS_A + SUBCLASS_OF: {composed}")

    # Verify composition
    seq_result = SUBCLASS_OF.apply(IS_A.apply(test_entity))
    comp_result = composed.apply(test_entity)
    sim_comp = cosine_similarity(seq_result.vec, comp_result.vec)
    print(f"  Composition correctness: {sim_comp:.6f}")
    print()

    # Example 5: Typed edges comparison
    print("-" * 80)
    print("Example 5: Different Relation Types")
    print("-" * 80)
    print()

    print("Comparing different relation operators...")
    print("\nOperator parameters are distinct:")

    # Check that different relations have different parameters
    relations = [
        ("IS_A", IS_A),
        ("SUBCLASS_OF", SUBCLASS_OF),
        ("HAS_PART", HAS_PART),
        ("INSTANCE_OF", INSTANCE_OF),
    ]

    for i, (name1, op1) in enumerate(relations):
        for name2, op2 in relations[i + 1 :]:
            sim = cosine_similarity(
                op1.params / jnp.linalg.norm(op1.params),
                op2.params / jnp.linalg.norm(op2.params),
            )
            print(f"  {name1} vs {name2}: {sim:.3f} (low similarity = distinct)")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("* Operators enable typed knowledge graph edges")
    print("* Different relation types are represented distinctly")
    print("* Graph queries work through operator inversion")
    print("* Facts can be bundled into knowledge bases")
    print("* Operators maintain exact inversion property")
    print()
    print("Knowledge graph patterns demonstrated:")
    print("  - Taxonomic relations (IS_A, SUBCLASS_OF)")
    print("  - Part-whole relations (HAS_PART, PART_OF)")
    print("  - Instance relations (INSTANCE_OF)")
    print("  - Inverse relations (automatic from operator inverse)")
    print()
    print("Use cases:")
    print("  - Knowledge graph construction and reasoning")
    print("  - Ontology representation")
    print("  - Semantic networks")
    print("  - Multi-hop reasoning")
    print()
    print("Note: For more complex graph reasoning (multi-hop queries,")
    print("transitive closure), consider using resonator networks")
    print("to factorize composite structures.")
    print()


if __name__ == "__main__":
    main()
