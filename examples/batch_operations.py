"""Example: Batch operations with JAX vmap for efficient computation.

This example demonstrates how to:
1. Use vmap_bind for batch binding
2. Use vmap_bundle for efficient bundling
3. Process large batches efficiently
4. Compare performance with individual operations
"""

import time

import jax
import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.utils import vmap_bind, vmap_bundle


def main() -> None:
    """Demonstrate batch operations with hypervectors."""
    print("=" * 60)
    print("VSAX Batch Operations Example")
    print("=" * 60)

    # Create FHRR model
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)

    # Example 1: Basic batch binding
    print("\n1. Basic batch binding...")

    # Add concepts
    nouns = ["dog", "cat", "bird", "fish"]
    verbs = ["runs", "jumps", "flies", "swims"]
    memory.add_many(nouns + verbs)

    # Batch bind nouns with verbs
    noun_vecs = jnp.stack([memory[n].vec for n in nouns])
    verb_vecs = jnp.stack([memory[v].vec for v in verbs])

    # Vectorized binding
    actions = vmap_bind(model.opset, noun_vecs, verb_vecs)

    print(f"   Bound {len(nouns)} noun-verb pairs")
    print(f"   Result shape: {actions.shape}")
    print(f"   Pairs: {list(zip(nouns, verbs))}")

    # Example 2: Batch bundling
    print("\n2. Batch bundling...")

    # Create a set of related concepts
    colors = ["red", "green", "blue", "yellow", "orange"]
    memory.add_many(colors)

    color_vecs = jnp.stack([memory[c].vec for c in colors])

    # Bundle all colors into a single "color" concept
    color_concept = vmap_bundle(model.opset, color_vecs)

    print(f"   Bundled {len(colors)} colors")
    print(f"   Result shape: {color_concept.shape}")

    # Example 3: Large batch processing
    print("\n3. Large batch processing...")

    # Create many symbols
    n_symbols = 100
    symbols = [f"sym_{i}" for i in range(n_symbols)]
    memory.add_many(symbols)

    # Create two large batches
    batch_size = 50
    X = jnp.stack([memory[f"sym_{i}"].vec for i in range(batch_size)])
    Y = jnp.stack([memory[f"sym_{i}"].vec for i in range(batch_size, 2 * batch_size)])

    print(f"   Processing {batch_size} bind operations...")

    # Time batch operation
    start = time.time()
    batch_result = vmap_bind(model.opset, X, Y)
    # Force computation
    jax.block_until_ready(batch_result)
    batch_time = time.time() - start

    print(f"   Batch vmap time: {batch_time:.4f}s")
    print(f"   Result shape: {batch_result.shape}")

    # Example 4: Sequential composition
    print("\n4. Sequential batch operations...")

    # Add role and filler concepts
    roles = ["subject", "verb", "object"]
    fillers = ["Alice", "likes", "Bob"]
    memory.add_many(roles + fillers)

    # First: bind roles with fillers
    role_vecs = jnp.stack([memory[r].vec for r in roles])
    filler_vecs = jnp.stack([memory[f].vec for f in fillers])

    bound_pairs = vmap_bind(model.opset, role_vecs, filler_vecs)
    print(f"   Bound {len(roles)} role-filler pairs")

    # Then: bundle all pairs into a sentence
    sentence = vmap_bundle(model.opset, bound_pairs)
    print(f"   Bundled into sentence vector, shape: {sentence.shape}")

    # Example 5: Nested structures
    print("\n5. Encoding nested structures...")

    # Build a simple knowledge graph
    # Relations: (subject, predicate, object)
    subjects = ["Alice", "Bob", "Charlie"]
    predicates = ["knows", "likes", "helps"]
    objects = ["Bob", "Alice", "Alice"]

    memory.add_many(list(set(subjects + predicates + objects)))

    # Encode each triple as bind(subject, bind(predicate, object))
    print("   Encoding triples:")
    for subj, pred, obj in zip(subjects, predicates, objects):
        print(f"     ({subj}, {pred}, {obj})")

    # Batch encode
    subj_vecs = jnp.stack([memory[s].vec for s in subjects])
    pred_vecs = jnp.stack([memory[p].vec for p in predicates])
    obj_vecs = jnp.stack([memory[o].vec for o in objects])

    # First bind predicates with objects
    pred_obj = vmap_bind(model.opset, pred_vecs, obj_vecs)

    # Then bind subjects with (predicate, object) pairs
    triples = vmap_bind(model.opset, subj_vecs, pred_obj)

    print(f"   Encoded {len(subjects)} triples")
    print(f"   Triple vectors shape: {triples.shape}")

    # Bundle all triples into a knowledge graph
    knowledge_graph = vmap_bundle(model.opset, triples)
    print(f"   Knowledge graph shape: {knowledge_graph.shape}")

    # Example 6: Hierarchical bundling
    print("\n6. Hierarchical bundling...")

    # Create a taxonomy
    mammals = ["dog", "cat", "elephant", "whale"]
    birds_list = ["eagle", "sparrow", "penguin"]
    reptiles = ["snake", "lizard", "crocodile"]

    all_animals = mammals + birds_list + reptiles
    memory.add_many(all_animals)

    # Bundle each category
    mammal_vecs = jnp.stack([memory[m].vec for m in mammals])
    mammal_concept = vmap_bundle(model.opset, mammal_vecs)

    bird_vecs = jnp.stack([memory[b].vec for b in birds_list])
    bird_concept = vmap_bundle(model.opset, bird_vecs)

    reptile_vecs = jnp.stack([memory[r].vec for r in reptiles])
    reptile_concept = vmap_bundle(model.opset, reptile_vecs)

    print(f"   Bundled {len(mammals)} mammals")
    print(f"   Bundled {len(birds_list)} birds")
    print(f"   Bundled {len(reptiles)} reptiles")

    # Bundle categories into "animal" concept
    category_vecs = jnp.stack([mammal_concept, bird_concept, reptile_concept])
    animal_concept = vmap_bundle(model.opset, category_vecs)

    print(f"   Created hierarchical 'animal' concept, shape: {animal_concept.shape}")

    # Example 7: Batch operations with different models
    print("\n7. Batch operations work across VSA models...")

    from vsax import create_binary_model, create_map_model

    models = [
        ("FHRR", create_fhrr_model(dim=256)),
        ("MAP", create_map_model(dim=256)),
        ("Binary", create_binary_model(dim=5000, bipolar=True)),
    ]

    for name, test_model in models:
        mem = VSAMemory(test_model)
        mem.add_many(["a", "b", "c", "x", "y", "z"])

        X = jnp.stack([mem["a"].vec, mem["b"].vec, mem["c"].vec])
        Y = jnp.stack([mem["x"].vec, mem["y"].vec, mem["z"].vec])

        result = vmap_bind(test_model.opset, X, Y)
        print(f"   {name:8s} - batch bind result shape: {result.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - vmap_bind: Efficient parallel binding of vector batches")
    print("  - vmap_bundle: Bundle multiple vectors efficiently")
    print("  - JAX vmap provides GPU/TPU acceleration")
    print("  - Works seamlessly with all VSA models (FHRR, MAP, Binary)")
    print("  - Essential for large-scale symbolic processing")
    print("=" * 60)


if __name__ == "__main__":
    main()
