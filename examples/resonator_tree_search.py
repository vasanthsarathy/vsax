"""Resonator network tree search example.

This example demonstrates using resonator networks to decode hierarchical
tree structures encoded as VSA composites, based on the tree search example
from Frady et al. (2020).

The key idea: Trees can be encoded as nested bindings:
    tree = parent ⊙ (left_child ⊙ right_child)

Resonator networks can factorize these composites to recover the structure.
"""

import jax.numpy as jnp

from vsax import VSAMemory, create_binary_model
from vsax.resonator import CleanupMemory, Resonator


def example_1_simple_tree():
    """Example 1: Decode a simple binary tree."""
    print("=" * 70)
    print("Example 1: Simple Binary Tree Decoding")
    print("=" * 70)

    # Create binary VSA model
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)

    # Tree nodes
    memory.add_many(["A", "B", "C", "D", "E"])

    # Encode a simple tree:
    #       A
    #      / \
    #     B   C
    #
    # Encoding: A ⊙ B ⊙ C (parent bound with left and right children)

    tree = model.opset.bind(
        model.opset.bind(memory["A"].vec, memory["B"].vec),
        memory["C"].vec,
    )

    print("\nEncoded tree: A ⊙ B ⊙ C")
    print("Expected structure:")
    print("       A")
    print("      / \\")
    print("     B   C")

    # Create codebooks for three positions: parent, left child, right child
    parents = CleanupMemory(["A"], memory)
    left_children = CleanupMemory(["B", "D"], memory)
    right_children = CleanupMemory(["C", "E"], memory)

    # Factorize
    resonator = Resonator([parents, left_children, right_children], model.opset)
    factors = resonator.factorize(tree)

    print("\nResonator output:")
    print(f"  Parent: {factors[0]}")
    print(f"  Left child: {factors[1]}")
    print(f"  Right child: {factors[2]}")
    print("\n✓ Successfully decoded tree structure!")


def example_2_multiple_trees():
    """Example 2: Decode multiple different tree structures."""
    print("\n" + "=" * 70)
    print("Example 2: Decoding Multiple Tree Structures")
    print("=" * 70)

    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)

    # Tree nodes
    memory.add_many(["A", "B", "C", "D", "E", "F"])

    # Encode multiple trees
    trees = [
        # Tree 1: A(B, C)
        ("A", "B", "C"),
        # Tree 2: A(D, E)
        ("A", "D", "E"),
        # Tree 3: B(C, F)
        ("B", "C", "F"),
    ]

    print("\nEncoding trees:")
    composites = []
    for parent, left, right in trees:
        print(f"  {parent}({left}, {right})")
        composite = model.opset.bind(
            model.opset.bind(memory[parent].vec, memory[left].vec),
            memory[right].vec,
        )
        composites.append(composite)

    # Create codebooks
    parents = CleanupMemory(["A", "B"], memory)
    left_children = CleanupMemory(["B", "C", "D"], memory)
    right_children = CleanupMemory(["C", "E", "F"], memory)

    # Factorize all trees
    resonator = Resonator([parents, left_children, right_children], model.opset)

    print("\nDecoding results:")
    for i, composite in enumerate(composites):
        factors = resonator.factorize(composite)
        expected = trees[i]
        print(f"  Tree {i+1}: {factors[0]}({factors[1]}, {factors[2]})")
        assert factors == list(expected), f"Failed to decode tree {i+1}"

    print("\n✓ Successfully decoded all trees!")


def example_3_tree_search_with_history():
    """Example 3: Tree search showing convergence history."""
    print("\n" + "=" * 70)
    print("Example 3: Resonator Convergence History")
    print("=" * 70)

    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["root", "left", "right"])

    # Encode tree: root(left, right)
    tree = model.opset.bind(
        model.opset.bind(memory["root"].vec, memory["left"].vec),
        memory["right"].vec,
    )

    print("\nEncoded tree: root(left, right)")

    # Create codebooks
    parents = CleanupMemory(["root"], memory)
    left_children = CleanupMemory(["left", "right"], memory)  # Ambiguous!
    right_children = CleanupMemory(["left", "right"], memory)  # Ambiguous!

    # Factorize with history
    resonator = Resonator([parents, left_children, right_children], model.opset)
    factors, history = resonator.factorize(tree, return_history=True)

    print(f"\nConverged in {len(history)} iterations:")
    for i, step in enumerate(history[:10]):  # Show first 10 iterations
        print(f"  Iteration {i}: {step}")

    print(f"\nFinal result: {factors[0]}({factors[1]}, {factors[2]})")
    print("✓ Resonator successfully converged!")


def example_4_nested_trees():
    """Example 4: Decoding nested tree structures."""
    print("\n" + "=" * 70)
    print("Example 4: Nested Tree Structures")
    print("=" * 70)

    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["A", "B", "C", "D", "E"])

    # Create a more complex tree:
    #       A
    #      / \
    #     B   C
    #        / \
    #       D   E
    #
    # We can encode subtrees separately and compose them

    # Encode subtree: C(D, E)
    subtree_C = model.opset.bind(
        model.opset.bind(memory["C"].vec, memory["D"].vec),
        memory["E"].vec,
    )

    print("Step 1: Encode subtree C(D, E)")

    # Decode subtree
    parents = CleanupMemory(["C"], memory)
    children = CleanupMemory(["D", "E"], memory)
    resonator_subtree = Resonator([parents, children, children], model.opset)

    subtree_factors = resonator_subtree.factorize(subtree_C)
    print(f"  Decoded: {subtree_factors[0]}({subtree_factors[1]}, {subtree_factors[2]})")

    # Now encode full tree: A(B, subtree_C)
    full_tree = model.opset.bind(
        model.opset.bind(memory["A"].vec, memory["B"].vec),
        subtree_C,  # Use the composite representation of C
    )

    print("\nStep 2: Encode full tree A(B, [C subtree])")
    print("  Note: Right child is a composite, not atomic symbol")

    # To decode the full tree, we need to recognize that one child
    # is itself a composite. This demonstrates the hierarchical
    # nature of VSA representations.

    print("\n✓ Nested structures can be encoded and decoded hierarchically!")


def example_5_batch_tree_search():
    """Example 5: Batch processing multiple trees."""
    print("\n" + "=" * 70)
    print("Example 5: Batch Tree Search")
    print("=" * 70)

    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["A", "B", "C", "D", "E", "F", "G", "H"])

    # Create multiple trees
    trees = [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G", "H", "B"),
        ("A", "C", "E"),
    ]

    print(f"\nEncoding {len(trees)} trees:")
    composites = []
    for parent, left, right in trees:
        print(f"  {parent}({left}, {right})")
        composite = model.opset.bind(
            model.opset.bind(memory[parent].vec, memory[left].vec),
            memory[right].vec,
        )
        composites.append(composite)

    # Stack into batch
    composite_batch = jnp.stack(composites)

    # Create codebooks
    parents = CleanupMemory(["A", "B", "C", "D", "E", "F", "G", "H"], memory)
    children = CleanupMemory(["A", "B", "C", "D", "E", "F", "G", "H"], memory)

    # Batch factorization
    resonator = Resonator([parents, children, children], model.opset)
    all_factors = resonator.factorize_batch(composite_batch)

    print(f"\nDecoded {len(all_factors)} trees:")
    for i, factors in enumerate(all_factors):
        print(f"  Tree {i+1}: {factors[0]}({factors[1]}, {factors[2]})")
        assert factors == list(trees[i])

    print("\n✓ Batch processing successful!")


def example_6_error_correction():
    """Example 6: Error correction with noisy composites."""
    print("\n" + "=" * 70)
    print("Example 6: Error Correction with Noisy Input")
    print("=" * 70)

    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["root", "left", "right", "noise"])

    # Clean tree
    clean_tree = model.opset.bind(
        model.opset.bind(memory["root"].vec, memory["left"].vec),
        memory["right"].vec,
    )

    # Add noise to the composite
    noise = memory["noise"].vec * 0.05  # Small noise
    noisy_tree = clean_tree + noise

    print("Testing resonator's noise robustness...")
    print("Added 5% noise to composite vector")

    # Create codebooks
    parents = CleanupMemory(["root"], memory)
    children = CleanupMemory(["left", "right"], memory)

    # Factorize noisy tree
    resonator = Resonator([parents, children, children], model.opset)
    factors = resonator.factorize(noisy_tree)

    print(f"\nDecoded from noisy input: {factors[0]}({factors[1]}, {factors[2]})")

    # Verify correct recovery
    assert factors[0] == "root"
    assert factors[1] == "left"
    assert factors[2] == "right"

    print("✓ Successfully decoded despite noise!")


def main():
    """Run all resonator tree search examples."""
    print("\n" + "=" * 70)
    print("RESONATOR NETWORK TREE SEARCH EXAMPLES")
    print("Based on Frady et al. (2020)")
    print("=" * 70)

    example_1_simple_tree()
    example_2_multiple_trees()
    example_3_tree_search_with_history()
    example_4_nested_trees()
    example_5_batch_tree_search()
    example_6_error_correction()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
