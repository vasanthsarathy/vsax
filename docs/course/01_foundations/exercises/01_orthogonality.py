"""
Module 1 Exercise 1: Exploring Orthogonality in High Dimensions

This exercise investigates how random vectors become orthogonal as dimension increases.

Tasks:
1. Generate random vectors in different dimensions
2. Measure pairwise cosine similarities
3. Plot the distribution of similarities
4. Find the "elbow" dimension where similarities drop below 0.05

Expected learning:
- Higher dimensions → more orthogonality
- Standard deviation decreases as 1/√d
- Practical dimension selection for VSA
"""

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np


def measure_orthogonality(dim, num_vectors=100, seed=0):
    """
    Generate random unit vectors and measure pairwise cosine similarities.

    Args:
        dim: Vector dimensionality
        num_vectors: Number of random vectors to generate
        seed: Random seed for reproducibility

    Returns:
        Array of pairwise similarities (off-diagonal elements)
    """
    key = random.PRNGKey(seed)

    # Generate random vectors
    vectors = random.normal(key, (num_vectors, dim))

    # Normalize to unit length
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Compute similarity matrix
    similarity_matrix = vectors @ vectors.T

    # Extract off-diagonal elements
    mask = ~jnp.eye(num_vectors, dtype=bool)
    off_diagonal = similarity_matrix[mask]

    return off_diagonal


def plot_similarity_distributions():
    """
    Plot how similarity distributions change with dimension.
    """
    dimensions = [10, 100, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, dim in enumerate(dimensions):
        similarities = measure_orthogonality(dim, num_vectors=500)

        axes[idx].hist(similarities, bins=50, alpha=0.7, edgecolor='black', density=True)
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
        axes[idx].set_title(f'Dimension = {dim}', fontsize=14)
        axes[idx].set_xlabel('Cosine Similarity', fontsize=12)
        axes[idx].set_ylabel('Density', fontsize=12)
        axes[idx].legend()
        axes[idx].set_xlim(-0.5, 0.5)

        # Add statistics
        mean_sim = float(jnp.mean(similarities))
        std_sim = float(jnp.std(similarities))
        axes[idx].text(
            0.05,
            0.95,
            f'μ={mean_sim:.3f}\nσ={std_sim:.3f}',
            transform=axes[idx].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig('similarity_distributions.png', dpi=150)
    print("Saved: similarity_distributions.png")
    plt.show()


def find_elbow_dimension():
    """
    Find the dimension where mean similarity drops below 0.05.
    """
    # Test logarithmically spaced dimensions
    dimensions = np.logspace(1, 4, num=30, dtype=int)  # 10 to 10,000
    mean_sims = []

    print("Testing dimensions...")
    for dim in dimensions:
        sims = measure_orthogonality(dim, num_vectors=100)
        mean_sim = float(jnp.mean(jnp.abs(sims)))
        mean_sims.append(mean_sim)
        print(f"  d={dim:5d}: mean |similarity| = {mean_sim:.4f}")

    # Find elbow (similarity < 0.05)
    threshold = 0.05
    elbow_indices = np.where(np.array(mean_sims) < threshold)[0]

    if len(elbow_indices) > 0:
        elbow_idx = elbow_indices[0]
        elbow_dim = dimensions[elbow_idx]
        print(f"\nElbow dimension (sim < {threshold}): d = {elbow_dim}")
    else:
        print(f"\nNo dimension found with similarity < {threshold}")

    # Plot scaling
    plt.figure(figsize=(10, 6))
    plt.loglog(dimensions, mean_sims, 'o-', linewidth=2, markersize=6, label='Empirical')
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    # Theoretical: 1/√d
    theoretical = [1 / np.sqrt(d) for d in dimensions]
    plt.loglog(dimensions, theoretical, '--', linewidth=2, alpha=0.7, label='Theoretical (1/√d)')

    plt.xlabel('Dimension (d)', fontsize=14)
    plt.ylabel('Mean Absolute Similarity', fontsize=14)
    plt.title('Orthogonality vs Dimension', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('dimension_scaling.png', dpi=150)
    print("Saved: dimension_scaling.png")
    plt.show()


def main():
    """
    Run all orthogonality experiments.
    """
    print("=" * 60)
    print("Exercise 1: Exploring Orthogonality in High Dimensions")
    print("=" * 60)

    print("\n[1] Testing different dimensions...")
    test_dims = [2, 10, 100, 1000, 10000]

    for dim in test_dims:
        sims = measure_orthogonality(dim)
        mean_sim = jnp.mean(jnp.abs(sims))
        std_sim = jnp.std(sims)
        theoretical_std = 1 / jnp.sqrt(dim)

        print(f"  Dimension {dim:5d}: mean |sim| = {mean_sim:.4f} ± {std_sim:.4f} (theory: {theoretical_std:.4f})")

    print("\n[2] Plotting similarity distributions...")
    plot_similarity_distributions()

    print("\n[3] Finding elbow dimension...")
    find_elbow_dimension()

    print("\n" + "=" * 60)
    print("Exercise complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Random vectors become orthogonal as dimension increases")
    print("- Standard deviation decreases as 1/√d")
    print("- For VSA, d=512-2048 provides good orthogonality")
    print("- Higher d = better separation but more computation")


if __name__ == "__main__":
    main()
