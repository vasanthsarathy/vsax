"""Spatial Semantic Pointers (SSP) - 1D Line Example.

This example demonstrates the basics of Spatial Semantic Pointers for
representing continuous positions on a 1D line. SSPs use fractional power
encoding to create continuous spatial representations.

Based on:
    Komer et al. 2019: "A neural representation of continuous space using
    fractional binding"
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vsax import VSAMemory, create_fhrr_model
from vsax.similarity import cosine_similarity
from vsax.spatial import SpatialSemanticPointers, SSPConfig


def main():
    """Demonstrate 1D spatial encoding with SSPs."""
    print("=" * 70)
    print("Spatial Semantic Pointers - 1D Line Example")
    print("=" * 70)

    # Initialize FHRR model and memory
    key = jax.random.PRNGKey(42)
    model = create_fhrr_model(dim=512, key=key)
    memory = VSAMemory(model)

    # Create 1D SSP encoder
    config = SSPConfig(dim=512, num_axes=1, scale=1.0, axis_names=["X"])
    ssp = SpatialSemanticPointers(model, memory, config)

    print("\n1. Encoding Spatial Positions")
    print("-" * 70)

    # Encode positions along the line
    positions = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    position_hvs = {}

    for pos in positions:
        pos_hv = ssp.encode_location([pos])
        position_hvs[pos] = pos_hv
        print(f"  Position {pos:.1f} encoded as hypervector (shape: {pos_hv.shape})")

    print("\n2. Similarity Between Positions")
    print("-" * 70)
    print("  Nearby positions should have higher similarity\n")

    # Compare position 2.0 with all others
    reference_pos = 2.0
    reference_hv = position_hvs[reference_pos]

    similarities = []
    for pos in positions:
        sim = cosine_similarity(reference_hv.vec, position_hvs[pos].vec)
        similarities.append(sim)
        print(f"  Position {reference_pos:.1f} vs {pos:.1f}: similarity = {sim:.4f}")

    print("\n3. Continuous Similarity Map")
    print("-" * 70)

    # Create a fine-grained similarity map
    query_positions = jnp.linspace(-1, 6, 100)
    similarity_map = []

    for q_pos in query_positions:
        q_hv = ssp.encode_location([float(q_pos)])
        sim = cosine_similarity(reference_hv.vec, q_hv.vec)
        similarity_map.append(sim)

    print(f"  Generated similarity map for {len(query_positions)} query points")
    print(f"  Reference position: {reference_pos:.1f}")

    # Plot similarity map
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(query_positions, similarity_map, linewidth=2)
    plt.axvline(reference_pos, color="r", linestyle="--", label=f"Reference ({reference_pos})")
    plt.xlabel("Position on Line")
    plt.ylabel("Similarity to Reference")
    plt.title("1D Spatial Similarity Map")
    plt.grid(True, alpha=0.3)
    plt.legend()

    print("\n4. Binding Objects to Locations")
    print("-" * 70)

    # Create object hypervectors
    objects = ["apple", "banana", "cherry"]
    for obj in objects:
        memory.add(obj)

    # Bind objects to specific positions
    scene_items = {
        "apple": 1.0,
        "banana": 3.0,
        "cherry": 5.0,
    }

    scene_hvs = []
    for obj, pos in scene_items.items():
        obj_loc_hv = ssp.bind_object_location(obj, [pos])
        scene_hvs.append(obj_loc_hv)
        print(f"  '{obj}' bound to position {pos:.1f}")

    # Bundle to create scene
    scene_vec = scene_hvs[0].vec
    for hv in scene_hvs[1:]:
        scene_vec = model.opset.bundle(scene_vec, hv.vec)

    from vsax.representations import ComplexHypervector
    scene = ComplexHypervector(scene_vec)

    print(f"\n  Scene created with {len(objects)} objects")

    print("\n5. Querying: What's at a Location?")
    print("-" * 70)

    # Query what's at each position
    for query_pos in [1.0, 3.0, 5.0]:
        result_hv = ssp.query_location(scene, [query_pos])

        # Find best match
        best_match = None
        best_sim = -1.0

        for obj in objects:
            obj_hv = memory[obj]
            sim = cosine_similarity(result_hv.vec, obj_hv.vec)
            if sim > best_sim:
                best_sim = sim
                best_match = obj

        print(f"  Position {query_pos:.1f}: '{best_match}' (similarity: {best_sim:.4f})")

    print("\n6. Querying: Where's an Object?")
    print("-" * 70)

    # Query where each object is located
    for obj in objects:
        result_hv = ssp.query_object(scene, obj)

        # Create similarity map to find peak
        position_sims = []
        test_positions = jnp.linspace(0, 6, 60)

        for test_pos in test_positions:
            test_hv = ssp.encode_location([float(test_pos)])
            sim = cosine_similarity(result_hv.vec, test_hv.vec)
            position_sims.append(sim)

        # Find peak
        peak_idx = jnp.argmax(jnp.array(position_sims))
        estimated_pos = test_positions[peak_idx]
        actual_pos = scene_items[obj]

        print(
            f"  '{obj}': estimated at {estimated_pos:.2f}, "
            f"actual at {actual_pos:.2f} (error: {abs(estimated_pos - actual_pos):.2f})"
        )

    # Plot where-is-object query for one object
    plt.subplot(1, 2, 2)
    test_obj = "banana"
    result_hv = ssp.query_object(scene, test_obj)

    position_sims = []
    test_positions = jnp.linspace(0, 6, 100)
    for test_pos in test_positions:
        test_hv = ssp.encode_location([float(test_pos)])
        sim = cosine_similarity(result_hv.vec, test_hv.vec)
        position_sims.append(sim)

    plt.plot(test_positions, position_sims, linewidth=2)
    plt.axvline(
        scene_items[test_obj],
        color="r",
        linestyle="--",
        label=f"Actual position ({scene_items[test_obj]})",
    )
    plt.xlabel("Position on Line")
    plt.ylabel("Similarity")
    plt.title(f"Where is '{test_obj}'?")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("ssp_1d_line.png", dpi=150, bbox_inches="tight")
    print("\n[OK] Plot saved to ssp_1d_line.png")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - SSPs represent continuous positions using fractional powers")
    print("  - Nearby positions have higher similarity")
    print("  - Objects can be bound to locations and queried")
    print("  - Both 'what' and 'where' queries are supported")
    print("=" * 70)


if __name__ == "__main__":
    main()
