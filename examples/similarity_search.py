"""Example: Similarity search using hypervectors.

This example demonstrates how to:
1. Build a memory of concepts
2. Query for similar concepts
3. Use different similarity metrics
4. Visualize search results
"""

import jax.numpy as jnp

from vsax import VSAMemory, create_fhrr_model
from vsax.similarity import cosine_similarity, dot_similarity
from vsax.utils import format_similarity_results, vmap_similarity


def main() -> None:
    """Demonstrate similarity search with hypervectors."""
    print("=" * 60)
    print("VSAX Similarity Search Example")
    print("=" * 60)

    # Create FHRR model
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)

    # Build a knowledge base of animals
    print("\n1. Building animal knowledge base...")
    animals = [
        # Mammals
        "dog",
        "cat",
        "wolf",
        "lion",
        "tiger",
        "elephant",
        "mouse",
        # Birds
        "eagle",
        "sparrow",
        "penguin",
        "parrot",
        # Reptiles
        "snake",
        "lizard",
        "crocodile",
        # Fish
        "shark",
        "goldfish",
        # Amphibians
        "frog",
        "salamander",
    ]
    memory.add_many(animals)
    print(f"   Added {len(animals)} animals to memory")

    # Add category concepts
    categories = ["mammal", "bird", "reptile", "fish", "amphibian", "predator", "pet"]
    memory.add_many(categories)
    print(f"   Added {len(categories)} categories")

    # Example 1: Find similar animals
    print("\n2. Finding animals similar to 'dog'...")
    query_name = "dog"
    query_vec = memory[query_name].vec

    # Get all animal vectors (exclude query)
    candidate_names = [name for name in animals if name != query_name]
    candidate_vecs = jnp.stack([memory[name].vec for name in candidate_names])

    # Compute similarities using vmap for efficiency (uses cosine internally)
    similarities = vmap_similarity(None, query_vec, candidate_vecs)

    # Format and display results
    results = format_similarity_results(
        query_name, candidate_names, similarities, top_k=5
    )
    print(results)

    # Example 2: Compare similarity metrics
    print("\n3. Comparing similarity metrics (dog vs wolf)...")
    dog_vec = memory["dog"].vec
    wolf_vec = memory["wolf"].vec

    cos_sim = cosine_similarity(dog_vec, wolf_vec)
    dot_sim = dot_similarity(dog_vec, wolf_vec)

    print(f"   Cosine similarity: {cos_sim:.4f}")
    print(f"   Dot similarity:    {dot_sim:.4f}")

    # Example 3: Multi-query search
    print("\n4. Finding similar concepts to 'cat'...")
    query_vec = memory["cat"].vec

    # Search across all concepts
    all_concepts = animals + categories
    candidate_names = [name for name in all_concepts if name != "cat"]
    candidate_vecs = jnp.stack([memory[name].vec for name in candidate_names])

    similarities = vmap_similarity(None, query_vec, candidate_vecs)

    results = format_similarity_results("cat", candidate_names, similarities, top_k=7)
    print(results)

    # Example 4: Category-based search
    print("\n5. Finding most 'predator'-like animals...")
    predator_vec = memory["predator"].vec

    # Search only among animals
    animal_vecs = jnp.stack([memory[name].vec for name in animals])
    similarities = vmap_similarity(cosine_similarity, predator_vec, animal_vecs)

    results = format_similarity_results("predator", animals, similarities, top_k=5)
    print(results)

    # Example 5: Composite query (combining concepts)
    print("\n6. Finding animals similar to 'predator + mammal'...")

    # Create composite query by bundling concepts
    predator_mammal = model.opset.bundle(memory["predator"].vec, memory["mammal"].vec)

    animal_vecs = jnp.stack([memory[name].vec for name in animals])
    similarities = vmap_similarity(cosine_similarity, predator_mammal, animal_vecs)

    results = format_similarity_results(
        "predator + mammal", animals, similarities, top_k=5
    )
    print(results)

    # Example 6: Similarity matrix
    print("\n7. Computing similarity matrix for select animals...")
    select_animals = ["dog", "wolf", "cat", "lion", "eagle", "snake"]

    print("\n   Similarity Matrix (Cosine):")
    print("   " + " " * 8, end="")
    for name in select_animals:
        print(f"{name:>8s}", end=" ")
    print()

    for name1 in select_animals:
        print(f"   {name1:>8s}", end=" ")
        vec1 = memory[name1].vec

        for name2 in select_animals:
            vec2 = memory[name2].vec
            sim = cosine_similarity(vec1, vec2)
            print(f"{sim:8.3f}", end=" ")
        print()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Similarity search enables finding related concepts")
    print("  - vmap_similarity provides efficient batch computation")
    print("  - Different metrics (cosine, dot) offer different perspectives")
    print("  - Composite queries can combine multiple concepts")
    print("=" * 60)


if __name__ == "__main__":
    main()
