"""Spatial Semantic Pointers (SSP) - 2D Navigation Example.

This example demonstrates using Spatial Semantic Pointers for representing
and reasoning about objects in a 2D environment. This reproduces concepts
from Komer et al. 2019.

Based on:
    Komer et al. 2019: "A neural representation of continuous space using
    fractional binding"
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vsax import VSAMemory, create_fhrr_model
from vsax.representations import ComplexHypervector
from vsax.similarity import cosine_similarity
from vsax.spatial import SpatialSemanticPointers, SSPConfig, similarity_map_2d


def main():
    """Demonstrate 2D spatial navigation with SSPs."""
    print("=" * 70)
    print("Spatial Semantic Pointers - 2D Navigation Example")
    print("=" * 70)

    # Initialize FHRR model with higher dimensionality for 2D
    key = jax.random.PRNGKey(42)
    model = create_fhrr_model(dim=1024, key=key)
    memory = VSAMemory(model)

    # Create 2D SSP encoder
    config = SSPConfig(dim=1024, num_axes=2, scale=1.0, axis_names=["X", "Y"])
    ssp = SpatialSemanticPointers(model, memory, config)

    print("\n1. Creating a 2D Environment")
    print("-" * 70)

    # Define landmarks in a 10x10 room
    landmarks = {
        "table": (3.0, 7.0),
        "chair": (5.0, 7.5),
        "plant": (8.0, 3.0),
        "lamp": (2.0, 2.5),
        "couch": (6.0, 5.0),
    }

    print("  Room size: 10x10 units")
    print("  Landmarks:")
    for obj, (x, y) in landmarks.items():
        memory.add(obj)  # Add object to memory
        print(f"    - {obj:8s} at ({x:.1f}, {y:.1f})")

    print("\n2. Encoding the Scene")
    print("-" * 70)

    # Create scene by binding objects to locations
    scene_hvs = []
    for obj, (x, y) in landmarks.items():
        obj_loc_hv = ssp.bind_object_location(obj, [x, y])
        scene_hvs.append(obj_loc_hv)

    # Bundle all object-location pairs
    scene_vec = scene_hvs[0].vec
    for hv in scene_hvs[1:]:
        scene_vec = model.opset.bundle(scene_vec, hv.vec)

    scene = ComplexHypervector(scene_vec)
    print(f"  Encoded {len(landmarks)} landmarks into scene hypervector")
    print(f"  Scene shape: {scene.shape}")

    print("\n3. Query: What's Near This Location?")
    print("-" * 70)

    # Query points
    query_points = [
        (3.0, 7.0),  # Near table
        (5.5, 7.2),  # Between table and chair
        (8.0, 3.0),  # Near plant
    ]

    for qx, qy in query_points:
        result_hv = ssp.query_location(scene, [qx, qy])

        print(f"\n  Querying position ({qx:.1f}, {qy:.1f}):")

        # Compare with all objects
        object_similarities = []
        for obj in landmarks.keys():
            obj_hv = memory[obj]
            sim = cosine_similarity(result_hv.vec, obj_hv.vec)
            object_similarities.append((obj, sim))

        # Sort by similarity
        object_similarities.sort(key=lambda x: x[1], reverse=True)

        for obj, sim in object_similarities[:3]:  # Top 3
            print(f"    {obj:8s}: {sim:6.4f}")

    print("\n4. Query: Where is Each Object?")
    print("-" * 70)

    # For each object, find where it is
    for obj, (true_x, true_y) in landmarks.items():
        result_hv = ssp.query_object(scene, obj)

        # Decode location using grid search
        decoded_loc = ssp.decode_location(
            result_hv, search_range=[(0, 10), (0, 10)], resolution=20
        )

        est_x, est_y = decoded_loc
        error = jnp.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2)

        print(
            f"  {obj:8s}: estimated ({est_x:5.2f}, {est_y:5.2f}), "
            f"actual ({true_x:4.1f}, {true_y:4.1f}), error: {error:.2f}"
        )

    print("\n5. Shifting the Scene")
    print("-" * 70)

    # Shift entire scene by (1, -1)
    dx, dy = 1.0, -1.0
    print(f"  Shifting scene by ({dx:.1f}, {dy:.1f})")

    shifted_scene = ssp.shift_scene(scene, [dx, dy])

    # Query shifted scene
    print("\n  Querying shifted scene at original landmark positions:")
    for obj, (orig_x, orig_y) in landmarks.items():
        # Query at shifted position
        shifted_x, shifted_y = orig_x + dx, orig_y + dy
        result_hv = ssp.query_location(shifted_scene, [shifted_x, shifted_y])

        # Should get back the same object
        obj_hv = memory[obj]
        sim = cosine_similarity(result_hv.vec, obj_hv.vec)

        print(f"    {obj:8s} at ({shifted_x:5.1f}, {shifted_y:5.1f}): similarity = {sim:.4f}")

    print("\n6. Generating Similarity Maps")
    print("-" * 70)

    # Create 2D similarity maps for "where is X?" queries
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    objects_to_plot = list(landmarks.keys())

    for idx, obj in enumerate(objects_to_plot):
        ax = axes[idx]

        # Query where object is
        result_hv = ssp.query_object(scene, obj)

        # Generate 2D similarity map
        X, Y, sim_map = similarity_map_2d(
            ssp, result_hv, x_range=(0, 10), y_range=(0, 10), resolution=50
        )

        # Plot heatmap
        im = ax.imshow(
            sim_map.T,
            extent=[0, 10, 0, 10],
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Where is '{obj}'?")

        # Mark actual location
        true_x, true_y = landmarks[obj]
        ax.plot(true_x, true_y, "b*", markersize=15, label="Actual")

        # Mark all landmarks
        for lm_name, (lm_x, lm_y) in landmarks.items():
            if lm_name != obj:
                ax.plot(lm_x, lm_y, "wo", markersize=8, alpha=0.5)
                ax.text(lm_x, lm_y - 0.3, lm_name, ha="center", fontsize=8, color="white")

        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, label="Similarity")

    # Hide the 6th subplot
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig("ssp_2d_navigation.png", dpi=150, bbox_inches="tight")
    print("  [OK] Similarity maps saved to ssp_2d_navigation.png")

    print("\n7. Environment Visualization")
    print("-" * 70)

    # Create environment visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all landmarks
    for obj, (x, y) in landmarks.items():
        ax.plot(x, y, "ro", markersize=20)
        ax.text(x, y + 0.3, obj, ha="center", fontsize=12, weight="bold")

    # Add grid
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("2D Navigation Environment", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.savefig("ssp_2d_environment.png", dpi=150, bbox_inches="tight")
    print("  [OK] Environment visualization saved to ssp_2d_environment.png")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - SSPs encode continuous 2D positions using X^x * Y^y")
    print("  - Multiple objects can be bundled into a single scene")
    print("  - Both 'what is at (x,y)' and 'where is X' queries work")
    print("  - Entire scenes can be shifted spatially")
    print("  - Similarity maps visualize spatial distributions")
    print("=" * 70)


if __name__ == "__main__":
    main()
