"""Example demonstrating save and load functionality for VSA basis vectors.

This example shows how to:
1. Create basis vectors and save them to JSON
2. Load basis vectors from JSON
3. Verify round-trip preservation
4. Handle all three VSA models (FHRR, MAP, Binary)
"""

from pathlib import Path

import jax.numpy as jnp

from vsax import (
    VSAMemory,
    create_binary_model,
    create_fhrr_model,
    create_map_model,
    load_basis,
    save_basis,
)
from vsax.similarity import cosine_similarity

print("=" * 70)
print("VSA Persistence Example: Save and Load Basis Vectors")
print("=" * 70)
print()

# =============================================================================
# Example 1: FHRR Model - Save and Load Complex Hypervectors
# =============================================================================
print("Example 1: FHRR Model (Complex Hypervectors)")
print("-" * 70)

# Create FHRR model and memory
fhrr_model = create_fhrr_model(dim=512)
memory_fhrr = VSAMemory(fhrr_model)

# Add semantic concepts
concepts = ["dog", "cat", "animal", "pet", "wild", "domestic"]
memory_fhrr.add_many(concepts)

print(f"Created {len(memory_fhrr._symbols)} FHRR vectors")

# Save to JSON
fhrr_path = Path("fhrr_basis.json")
save_basis(memory_fhrr, fhrr_path)
print(f"✓ Saved to {fhrr_path}")

# Load into new memory
memory_fhrr_loaded = VSAMemory(fhrr_model)
load_basis(memory_fhrr_loaded, fhrr_path)
print(f"✓ Loaded {len(memory_fhrr_loaded._symbols)} vectors")

# Verify vectors are identical
for name in concepts:
    original = memory_fhrr[name].vec
    loaded = memory_fhrr_loaded[name].vec
    assert jnp.allclose(original, loaded, atol=1e-6)

print("✓ All vectors preserved exactly!")
print()

# Clean up
fhrr_path.unlink()

# =============================================================================
# Example 2: MAP Model - Save and Load Real Hypervectors
# =============================================================================
print("Example 2: MAP Model (Real Hypervectors)")
print("-" * 70)

# Create MAP model and memory
map_model = create_map_model(dim=512)
memory_map = VSAMemory(map_model)

# Add colors
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
memory_map.add_many(colors)

print(f"Created {len(memory_map._symbols)} MAP vectors")

# Save to JSON
map_path = Path("map_basis.json")
save_basis(memory_map, map_path)
print(f"✓ Saved to {map_path}")

# Load into new memory
memory_map_loaded = VSAMemory(map_model)
load_basis(memory_map_loaded, map_path)
print(f"✓ Loaded {len(memory_map_loaded._symbols)} vectors")

# Verify preservation
for name in colors:
    original = memory_map[name].vec
    loaded = memory_map_loaded[name].vec
    assert jnp.allclose(original, loaded, atol=1e-6)

print("✓ All vectors preserved exactly!")
print()

# Clean up
map_path.unlink()

# =============================================================================
# Example 3: Binary Model - Save and Load Bipolar Hypervectors
# =============================================================================
print("Example 3: Binary Model (Bipolar Hypervectors)")
print("-" * 70)

# Create Binary model and memory
binary_model = create_binary_model(dim=10000, bipolar=True)
memory_binary = VSAMemory(binary_model)

# Add programming languages
languages = ["python", "java", "rust", "go", "javascript", "typescript"]
memory_binary.add_many(languages)

print(f"Created {len(memory_binary._symbols)} Binary vectors")

# Save to JSON
binary_path = Path("binary_basis.json")
save_basis(memory_binary, binary_path)
print(f"✓ Saved to {binary_path}")

# Load into new memory
memory_binary_loaded = VSAMemory(binary_model)
load_basis(memory_binary_loaded, binary_path)
print(f"✓ Loaded {len(memory_binary_loaded._symbols)} vectors")

# Verify preservation
for name in languages:
    original = memory_binary[name].vec
    loaded = memory_binary_loaded[name].vec
    assert jnp.allclose(original, loaded)

print("✓ All vectors preserved exactly!")
print()

# Clean up
binary_path.unlink()

# =============================================================================
# Example 4: Practical Use Case - Persistent Semantic Space
# =============================================================================
print("Example 4: Practical Use Case - Persistent Semantic Space")
print("-" * 70)

# Create a semantic space for a knowledge base
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Build semantic concepts
entities = ["alice", "bob", "charlie"]
actions = ["loves", "knows", "helps"]
objects = ["book", "music", "art"]

all_symbols = entities + actions + objects
memory.add_many(all_symbols)

print(f"Created semantic space with {len(all_symbols)} symbols")

# Create some semantic relationships
alice_loves_music = model.opset.bind(
    model.opset.bind(memory["alice"].vec, memory["loves"].vec), memory["music"].vec
)

# Save the basis for later use
semantic_path = Path("semantic_space.json")
save_basis(memory, semantic_path)
print(f"✓ Saved semantic space to {semantic_path}")

# Simulate later session - load the space
print("\n--- Later Session ---")
memory_new = VSAMemory(model)
load_basis(memory_new, semantic_path)
print(f"✓ Loaded {len(memory_new._symbols)} symbols")

# Reconstruct the same relationship
alice_loves_music_reconstructed = model.opset.bind(
    model.opset.bind(memory_new["alice"].vec, memory_new["loves"].vec),
    memory_new["music"].vec,
)

# Verify relationships are preserved
assert jnp.allclose(alice_loves_music, alice_loves_music_reconstructed, atol=1e-6)
print("✓ Semantic relationships preserved across sessions!")

# Clean up
semantic_path.unlink()
print()

# =============================================================================
# Example 5: Sharing Basis Vectors Between Projects
# =============================================================================
print("Example 5: Sharing Basis Vectors Between Projects")
print("-" * 70)

# Project 1: Create a shared vocabulary
project1_model = create_map_model(dim=512)
project1_memory = VSAMemory(project1_model)

shared_vocab = [
    "function",
    "class",
    "method",
    "variable",
    "parameter",
    "return",
    "import",
    "export",
]
project1_memory.add_many(shared_vocab)

# Save for sharing
shared_path = Path("shared_vocabulary.json")
save_basis(project1_memory, shared_path)
print(f"Project 1: Saved {len(shared_vocab)} terms to {shared_path}")

# Project 2: Load the shared vocabulary
project2_model = create_map_model(dim=512)  # Must match dimension!
project2_memory = VSAMemory(project2_model)
load_basis(project2_memory, shared_path)

print(f"Project 2: Loaded {len(project2_memory._symbols)} shared terms")

# Both projects can now use identical basis vectors
sim = cosine_similarity(project1_memory["function"], project2_memory["function"])
print(f"✓ Similarity between shared 'function' vectors: {sim:.6f}")
assert sim > 0.999  # Should be nearly identical

# Clean up
shared_path.unlink()
print()

# =============================================================================
# Example 6: Error Handling
# =============================================================================
print("Example 6: Error Handling")
print("-" * 70)

# Save with one model
model_128 = create_fhrr_model(dim=128)
memory_128 = VSAMemory(model_128)
memory_128.add("test")

test_path = Path("test_basis.json")
save_basis(memory_128, test_path)
print("✓ Saved 128-dim FHRR basis")

# Try to load with wrong dimension
model_256 = create_fhrr_model(dim=256)
memory_256 = VSAMemory(model_256)

try:
    load_basis(memory_256, test_path)
    print("❌ Should have raised dimension mismatch error")
except ValueError as e:
    print(f"✓ Caught dimension mismatch: {e}")

# Try to load with wrong representation type
map_model_128 = create_map_model(dim=128)
memory_map_128 = VSAMemory(map_model_128)

try:
    load_basis(memory_map_128, test_path)
    print("❌ Should have raised rep type mismatch error")
except ValueError as e:
    print(f"✓ Caught rep type mismatch: {e}")

# Try to load into non-empty memory
memory_nonempty = VSAMemory(model_128)
memory_nonempty.add("existing")

try:
    load_basis(memory_nonempty, test_path)
    print("❌ Should have raised non-empty memory error")
except ValueError as e:
    print(f"✓ Caught non-empty memory error: {e}")

# Clean up
test_path.unlink()
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("✓ Save/load works for all three VSA models (FHRR, MAP, Binary)")
print("✓ Vectors are preserved exactly in round-trip")
print("✓ JSON format is human-readable and portable")
print("✓ Validation prevents mismatched loads")
print("✓ Enables sharing basis vectors between projects")
print("✓ Perfect for persistent semantic spaces and knowledge bases")
print()
print("Key Functions:")
print("  - save_basis(memory, path)  # Save vectors to JSON")
print("  - load_basis(memory, path)  # Load vectors from JSON")
print()
