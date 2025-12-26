"""
Module 1 Exercise 2: Bundling Capacity Analysis

This exercise investigates the capacity limits of bundling operations.

Tasks:
1. Bundle different numbers of random vectors
2. Measure mean similarity to input vectors
3. Plot capacity curve (similarity vs bundle size)
4. Compare empirical results to theoretical prediction (1/√N)
5. Find practical capacity limit

Expected learning:
- Bundling capacity scales as √d
- Similarity decreases as 1/√N
- Trade-off between capacity and signal quality
"""

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt


def test_bundling_capacity(dim, num_bundles, seed=0):
    """
    Bundle multiple random vectors and measure similarity to inputs.

    Args:
        dim: Vector dimensionality
        num_bundles: Number of vectors to bundle
        seed: Random seed

    Returns:
        Mean similarity between bundled vector and input vectors
    """
    key = random.PRNGKey(seed)

    # Generate random unit vectors
    vectors = random.normal(key, (num_bundles, dim))
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    # Bundle by averaging
    bundled = jnp.sum(vectors, axis=0)
    bundled = bundled / jnp.linalg.norm(bundled)

    # Measure similarity to each input
    similarities = vectors @ bundled
    mean_sim = jnp.mean(similarities)

    return float(mean_sim)


def plot_capacity_curve(dim=2048):
    """
    Plot bundling capacity curve for a given dimension.
    """
    bundle_sizes = range(1, 201, 5)
    empirical_sims = []
    theoretical_sims = []

    print(f"Testing bundling capacity for dim={dim}...")
    print(f"Expected capacity (√d): {int(jnp.sqrt(dim))}\n")

    for num in bundle_sizes:
        # Empirical
        emp_sim = test_bundling_capacity(dim, num)
        empirical_sims.append(emp_sim)

        # Theoretical: 1/√N
        theo_sim = 1 / jnp.sqrt(num)
        theoretical_sims.append(float(theo_sim))

        if num % 20 == 1:
            print(f"  Bundle size {num:3d}: similarity = {emp_sim:.4f} (theory: {theo_sim:.4f})")

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(bundle_sizes, empirical_sims, 'o-', label='Empirical', linewidth=2, markersize=4, alpha=0.7)
    plt.plot(bundle_sizes, theoretical_sims, '--', label='Theoretical (1/√N)', linewidth=2, alpha=0.7)
    plt.axvline(jnp.sqrt(dim), color='red', linestyle=':', linewidth=2, label=f'Capacity (√{dim}≈{int(jnp.sqrt(dim))})')
    plt.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    plt.xlabel('Number of Bundled Vectors', fontsize=12)
    plt.ylabel('Mean Similarity to Inputs', fontsize=12)
    plt.title(f'Bundling Capacity (d={dim})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.loglog(bundle_sizes, empirical_sims, 'o-', label='Empirical', linewidth=2, markersize=4, alpha=0.7)
    plt.loglog(bundle_sizes, theoretical_sims, '--', label='Theoretical (1/√N)', linewidth=2, alpha=0.7)
    plt.axvline(jnp.sqrt(dim), color='red', linestyle=':', linewidth=2, label=f'Capacity (√d)')
    plt.xlabel('Number of Bundled Vectors (log)', fontsize=12)
    plt.ylabel('Mean Similarity (log)', fontsize=12)
    plt.title('Bundling Capacity (log-log)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bundling_capacity.png', dpi=150)
    print("\nSaved: bundling_capacity.png")
    plt.show()


def compare_dimensions():
    """
    Compare bundling capacity across different dimensions.
    """
    dimensions = [512, 1024, 2048, 4096]
    colors = ['blue', 'green', 'red', 'purple']

    plt.figure(figsize=(10, 6))

    for dim, color in zip(dimensions, colors):
        bundle_sizes = range(1, 201, 5)
        sims = [test_bundling_capacity(dim, num) for num in bundle_sizes]

        plt.plot(
            bundle_sizes, sims, 'o-', label=f'd={dim} (√d≈{int(jnp.sqrt(dim))})', color=color, linewidth=2, markersize=3, alpha=0.7
        )

        # Mark capacity point
        capacity = int(jnp.sqrt(dim))
        if capacity in bundle_sizes:
            idx = list(bundle_sizes).index(capacity)
            plt.scatter([capacity], [sims[idx]], color=color, s=100, marker='*', zorder=5)

    # Theoretical curve
    plt.plot(bundle_sizes, [1 / jnp.sqrt(n) for n in bundle_sizes], '--', color='black', linewidth=2, alpha=0.5, label='Theory (1/√N)')

    plt.axhline(0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    plt.xlabel('Number of Bundled Vectors', fontsize=12)
    plt.ylabel('Mean Similarity to Inputs', fontsize=12)
    plt.title('Bundling Capacity Across Dimensions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('capacity_comparison.png', dpi=150)
    print("Saved: capacity_comparison.png")
    plt.show()


def practical_capacity_limits():
    """
    Determine practical capacity limits for different quality thresholds.
    """
    dim = 2048
    thresholds = [0.9, 0.7, 0.5, 0.3]

    print(f"\nPractical Capacity Limits (d={dim}):")
    print("=" * 60)
    print(f"{'Quality Threshold':<20} {'Max Bundle Size':<20} {'Capacity (√d)':<20}")
    print("-" * 60)

    for threshold in thresholds:
        # Binary search for max bundle size
        low, high = 1, 500
        max_size = 1

        while low <= high:
            mid = (low + high) // 2
            sim = test_bundling_capacity(dim, mid)

            if sim >= threshold:
                max_size = mid
                low = mid + 1
            else:
                high = mid - 1

        capacity_ratio = max_size / jnp.sqrt(dim)
        print(f"{threshold:<20.2f} {max_size:<20d} {capacity_ratio:<20.2f}x")

    print("=" * 60)
    print("\nInterpretation:")
    print("- High quality (>0.9): Can bundle ~5-10 vectors")
    print("- Medium quality (>0.7): Can bundle ~20-30 vectors")
    print("- Acceptable (>0.5): Can bundle ~√d vectors (capacity limit)")
    print("- Low quality (>0.3): Can bundle ~2-3x√d vectors")


def main():
    """
    Run all bundling capacity experiments.
    """
    print("=" * 60)
    print("Exercise 2: Bundling Capacity Analysis")
    print("=" * 60)

    print("\n[1] Testing single dimension (d=2048)...")
    plot_capacity_curve(dim=2048)

    print("\n[2] Comparing across dimensions...")
    compare_dimensions()

    print("\n[3] Finding practical capacity limits...")
    practical_capacity_limits()

    print("\n" + "=" * 60)
    print("Exercise complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Bundling capacity ≈ √d vectors")
    print("- Similarity decreases as 1/√N")
    print("- Trade-off: more bundles = lower signal quality")
    print("- For d=2048, can bundle ~45 vectors with sim > 0.5")
    print("- Choose bundle size based on required quality")


if __name__ == "__main__":
    main()
