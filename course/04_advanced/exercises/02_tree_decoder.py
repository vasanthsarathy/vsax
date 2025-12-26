"""
Module 4 Exercise 2: Tree Decoder with Resonator Networks

This exercise combines hierarchical structure encoding (Lesson 4.3) with
resonator networks for factorization to build and decode complex tree structures.

Tasks:
1. Build encoders for various tree types (expression trees, parse trees)
2. Encode complex nested structures
3. Use resonator networks to decode factorized compositions
4. Handle variable-depth hierarchies
5. Test on real-world examples (math expressions, JSON, family trees)

Expected learning:
- Recursive role-filler binding patterns
- Resonator-based factorization for nested structures
- Handling variable depth and complexity
- Practical applications for parsing and data structures
"""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity
from vsax.resonator import CleanupMemory, Resonator
from typing import Any, Dict, List, Optional, Union
import numpy as np


class TreeNode:
    """
    Simple tree node for representing hierarchical structures.
    """

    def __init__(self, value, children=None):
        """
        Create tree node.

        Args:
            value: Node value (str or number)
            children: List of child TreeNode objects
        """
        self.value = value
        self.children = children or []

    def is_leaf(self):
        """Check if node is a leaf."""
        return len(self.children) == 0

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf({self.value})"
        return f"Node({self.value}, {len(self.children)} children)"


class HierarchicalEncoder:
    """
    Encodes hierarchical tree structures using recursive role-filler binding.
    """

    def __init__(self, model, memory, max_children=5):
        """
        Initialize hierarchical encoder.

        Args:
            model: VSA model
            memory: VSA memory
            max_children: Maximum number of children per node
        """
        self.model = model
        self.memory = memory
        self.max_children = max_children

        # Add role vectors
        roles = ["value", "op", "operator"]
        roles += [f"child_{i}" for i in range(max_children)]
        self.memory.add_many(roles)

    def encode_leaf(self, value):
        """
        Encode leaf node.

        Args:
            value: Leaf value

        Returns:
            Hypervector for leaf
        """
        value_str = str(value)
        if value_str not in self.memory:
            self.memory.add(value_str)
        return self.memory[value_str].vec

    def encode_node(self, value, children_hvs):
        """
        Encode internal node with children.

        Args:
            value: Node value (operator, type, etc.)
            children_hvs: List of child hypervectors

        Returns:
            Hypervector for node
        """
        # Encode value
        value_str = str(value)
        if value_str not in self.memory:
            self.memory.add(value_str)

        value_hv = self.model.opset.bind(
            self.memory["value"].vec,
            self.memory[value_str].vec
        )

        # Bind each child with its role
        components = [value_hv]
        for i, child_hv in enumerate(children_hvs):
            if i >= self.max_children:
                print(f"Warning: Node has more than {self.max_children} children")
                break

            child_role = self.memory[f"child_{i}"].vec
            child_binding = self.model.opset.bind(child_role, child_hv)
            components.append(child_binding)

        # Bundle all components
        return self.model.opset.bundle(*components)

    def encode_tree(self, node):
        """
        Recursively encode entire tree.

        Args:
            node: TreeNode object

        Returns:
            Hypervector representing entire tree
        """
        if node.is_leaf():
            return self.encode_leaf(node.value)
        else:
            # Recursively encode children
            child_hvs = [self.encode_tree(child) for child in node.children]
            return self.encode_node(node.value, child_hvs)


class HierarchicalDecoder:
    """
    Decodes hierarchical structures using unbinding and resonators.
    """

    def __init__(self, model, memory, encoder):
        """
        Initialize decoder.

        Args:
            model: VSA model
            memory: VSA memory
            encoder: HierarchicalEncoder instance
        """
        self.model = model
        self.memory = memory
        self.encoder = encoder

    def decode_value(self, node_hv, candidates):
        """
        Decode node value by unbinding and cleanup.

        Args:
            node_hv: Node hypervector
            candidates: List of possible values

        Returns:
            (best_match, similarity)
        """
        # Unbind value role
        value_hv = self.model.opset.bind(
            node_hv,
            self.model.opset.inverse(self.memory["value"].vec)
        )

        # Find best match
        best_match = None
        best_sim = -1.0

        for candidate in candidates:
            if str(candidate) in self.memory:
                sim = cosine_similarity(value_hv, self.memory[str(candidate)].vec)
                if sim > best_sim:
                    best_sim = sim
                    best_match = candidate

        return best_match, best_sim

    def decode_child(self, node_hv, child_idx):
        """
        Decode specific child by unbinding.

        Args:
            node_hv: Parent node hypervector
            child_idx: Child index (0, 1, 2, ...)

        Returns:
            Child hypervector
        """
        child_role = self.memory[f"child_{child_idx}"].vec
        child_hv = self.model.opset.bind(
            node_hv,
            self.model.opset.inverse(child_role)
        )
        return child_hv

    def decode_tree(self, encoded_hv, value_candidates, leaf_candidates,
                    max_depth=10):
        """
        Recursively decode tree structure.

        Args:
            encoded_hv: Encoded tree hypervector
            value_candidates: Possible internal node values
            leaf_candidates: Possible leaf values
            max_depth: Maximum recursion depth

        Returns:
            Reconstructed TreeNode
        """
        if max_depth == 0:
            return TreeNode("MAX_DEPTH_EXCEEDED")

        # Try to decode as leaf
        leaf_value, leaf_sim = self.decode_value(encoded_hv, leaf_candidates)

        # Try to decode as internal node
        node_value, node_sim = self.decode_value(encoded_hv, value_candidates)

        # Decide if leaf or internal
        if leaf_sim > node_sim and leaf_sim > 0.5:
            return TreeNode(leaf_value)
        elif node_sim > 0.5:
            # Internal node - decode children
            children = []
            for child_idx in range(self.encoder.max_children):
                child_hv = self.decode_child(encoded_hv, child_idx)

                # Check if child exists (has meaningful value)
                child_leaf, child_leaf_sim = self.decode_value(child_hv, leaf_candidates)
                child_node, child_node_sim = self.decode_value(child_hv, value_candidates)

                if max(child_leaf_sim, child_node_sim) > 0.4:
                    # Child exists, decode recursively
                    child_tree = self.decode_tree(
                        child_hv, value_candidates, leaf_candidates, max_depth - 1
                    )
                    children.append(child_tree)
                else:
                    # No more children
                    break

            return TreeNode(node_value, children)
        else:
            # Unknown
            return TreeNode("UNKNOWN")


def test_expression_trees():
    """
    Test encoding/decoding arithmetic expression trees.
    """
    print("=" * 70)
    print("Test 1: Arithmetic Expression Trees")
    print("=" * 70)

    # Create model
    model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(42))
    memory = VSAMemory(model)

    encoder = HierarchicalEncoder(model, memory, max_children=3)
    decoder = HierarchicalDecoder(model, memory, encoder)

    # Build expression tree: ((2 + 3) * 4)
    print("\n--- Building expression: ((2 + 3) * 4) ---")

    leaf_2 = TreeNode("2")
    leaf_3 = TreeNode("3")
    leaf_4 = TreeNode("4")

    add_node = TreeNode("+", [leaf_2, leaf_3])
    mult_node = TreeNode("*", [add_node, leaf_4])

    # Encode
    encoded = encoder.encode_tree(mult_node)
    print(f"Encoded expression as {model.dim}-dimensional vector")

    # Decode
    operators = ["+", "*", "-", "/"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    decoded = decoder.decode_tree(encoded, operators, numbers)

    # Print results
    def print_tree(node, indent=0):
        prefix = "  " * indent
        if node.is_leaf():
            print(f"{prefix}Leaf: {node.value}")
        else:
            print(f"{prefix}Node: {node.value}")
            for child in node.children:
                print_tree(child, indent + 1)

    print("\nDecoded tree:")
    print_tree(decoded)

    # Verify
    print("\nVerification:")
    print(f"Root operator: {decoded.value} (expected: *)")
    if len(decoded.children) > 0:
        print(f"Left child: {decoded.children[0].value} (expected: +)")
    if len(decoded.children) > 1:
        print(f"Right child: {decoded.children[1].value} (expected: 4)")

    print("\n" + "=" * 70)


def test_nested_json():
    """
    Test encoding/decoding nested JSON-like structures.
    """
    print("\n" + "=" * 70)
    print("Test 2: Nested JSON-like Structures")
    print("=" * 70)

    model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(123))
    memory = VSAMemory(model)

    encoder = HierarchicalEncoder(model, memory, max_children=4)
    decoder = HierarchicalDecoder(model, memory, encoder)

    # Build tree: {"person": {"name": "Alice", "age": "30"}}
    print("\n--- Building JSON: {person: {name: Alice, age: 30}} ---")

    name_value = TreeNode("Alice")
    age_value = TreeNode("30")

    name_pair = TreeNode("name", [name_value])
    age_pair = TreeNode("age", [age_value])

    person_obj = TreeNode("person", [name_pair, age_pair])

    # Encode
    encoded = encoder.encode_tree(person_obj)
    print(f"Encoded JSON as {model.dim}-dimensional vector")

    # Decode
    keys = ["person", "name", "age", "city"]
    values = ["Alice", "Bob", "Charlie", "30", "25", "NYC"]

    decoded = decoder.decode_tree(encoded, keys, values)

    # Print
    def print_json_tree(node, indent=0):
        prefix = "  " * indent
        if node.is_leaf():
            print(f"{prefix}\"{node.value}\"")
        else:
            print(f"{prefix}{node.value}: {{")
            for child in node.children:
                print_json_tree(child, indent + 1)
            print(f"{prefix}}}")

    print("\nDecoded structure:")
    print_json_tree(decoded)

    print("\n" + "=" * 70)


def test_resonator_factorization():
    """
    Test using resonator to factorize complex bindings.
    """
    print("\n" + "=" * 70)
    print("Test 3: Resonator-Based Factorization")
    print("=" * 70)

    model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(456))
    memory = VSAMemory(model)

    # Create composite: operator ⊗ left_child ⊗ right_child
    memory.add_many(["+", "*", "2", "3", "4", "5"])

    print("\n--- Creating composite: + ⊗ 2 ⊗ 3 ---")

    composite = model.opset.bind(
        model.opset.bind(memory["+"].vec, memory["2"].vec),
        memory["3"].vec
    )

    # Create codebooks
    operators = CleanupMemory(["+", "*", "-", "/"], memory)
    numbers1 = CleanupMemory(["2", "3", "4", "5"], memory)
    numbers2 = CleanupMemory(["2", "3", "4", "5"], memory)

    # Create resonator
    resonator = Resonator(
        codebooks=[operators, numbers1, numbers2],
        opset=model.opset,
        max_iterations=50,
        convergence_threshold=0.95
    )

    # Factorize
    print("Running resonator factorization...")
    factors = resonator.factorize(composite)

    print(f"\nRecovered factors: {factors}")
    print(f"Expected: ['+', '2', '3']")

    # Verify
    reconstructed = model.opset.bind(
        model.opset.bind(memory[factors[0]].vec, memory[factors[1]].vec),
        memory[factors[2]].vec
    )
    sim = cosine_similarity(reconstructed, composite)
    print(f"Reconstruction similarity: {sim:.3f}")

    print("\n" + "=" * 70)


def test_variable_depth_trees():
    """
    Test trees of varying depth.
    """
    print("\n" + "=" * 70)
    print("Test 4: Variable Depth Trees")
    print("=" * 70)

    model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(789))
    memory = VSAMemory(model)

    encoder = HierarchicalEncoder(model, memory, max_children=3)
    decoder = HierarchicalDecoder(model, memory, encoder)

    # Test different depths
    depths = [1, 2, 3, 4]

    def build_chain(depth):
        """Build chain tree of given depth."""
        if depth == 0:
            return TreeNode("leaf")
        else:
            return TreeNode("node", [build_chain(depth - 1)])

    print("\n--- Testing trees of depth 1 to 4 ---")

    operators = ["node"]
    leaves = ["leaf"]

    for depth in depths:
        tree = build_chain(depth)
        encoded = encoder.encode_tree(tree)
        decoded = decoder.decode_tree(encoded, operators, leaves)

        # Count depth
        def tree_depth(node):
            if node.is_leaf():
                return 0
            elif len(node.children) == 0:
                return 0
            else:
                return 1 + max(tree_depth(child) for child in node.children)

        decoded_depth = tree_depth(decoded)
        print(f"Depth {depth}: Encoded & decoded (recovered depth: {decoded_depth})")

        if decoded_depth == depth:
            print(f"  ✓ Correct depth recovered")
        else:
            print(f"  ✗ Depth mismatch")

    print("\n" + "=" * 70)


def test_family_tree():
    """
    Test encoding family tree relationships.
    """
    print("\n" + "=" * 70)
    print("Test 5: Family Tree Encoding")
    print("=" * 70)

    model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(999))
    memory = VSAMemory(model)

    encoder = HierarchicalEncoder(model, memory, max_children=5)
    decoder = HierarchicalDecoder(model, memory, encoder)

    # Build family tree
    # Alice (root) -> Bob, Charlie
    # Bob -> Diana, Emma
    print("\n--- Building family tree ---")

    diana = TreeNode("Diana")
    emma = TreeNode("Emma")
    bob = TreeNode("Bob", [diana, emma])

    charlie = TreeNode("Charlie")

    alice = TreeNode("Alice", [bob, charlie])

    # Encode
    encoded = encoder.encode_tree(alice)
    print(f"Encoded family tree")

    # Decode
    people = ["Alice", "Bob", "Charlie", "Diana", "Emma"]
    decoded = decoder.decode_tree(encoded, people, people)

    # Print
    def print_family(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{node.value}")
        for child in node.children:
            print_family(child, indent + 1)

    print("\nDecoded family tree:")
    print_family(decoded)

    # Verify structure
    print("\nVerification:")
    print(f"Root: {decoded.value} (expected: Alice)")
    if len(decoded.children) >= 1:
        print(f"First child: {decoded.children[0].value} (expected: Bob)")
        if len(decoded.children[0].children) >= 1:
            print(f"Bob's first child: {decoded.children[0].children[0].value} (expected: Diana)")

    print("\n" + "=" * 70)


def main():
    """
    Run all tree decoder tests.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 4 EXERCISE 2")
    print(" " * 22 + "Tree Decoder with Resonators")
    print("=" * 80)

    test_expression_trees()
    test_nested_json()
    test_resonator_factorization()
    test_variable_depth_trees()
    test_family_tree()

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("✓ Recursive role-filler binding encodes arbitrary tree structures")
    print("✓ Unbinding with cleanup decodes nodes and children")
    print("✓ Resonators factorize complex multi-factor bindings")
    print("✓ Variable depth trees can be handled with max_depth limits")
    print("✓ Applications: parse trees, JSON, family trees, expression trees")
    print("=" * 80)


if __name__ == "__main__":
    main()
