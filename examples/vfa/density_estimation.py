"""Vector Function Architecture - Density Estimation Example.

This example demonstrates kernel density estimation using VFA, reproducing
concepts from Frady et al. 2021 §7.2.1.

Based on:
    Frady et al. 2021: "Computing on Functions Using Randomized Vector
    Representations"
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vsax import VSAMemory, create_fhrr_model
from vsax.vfa import DensityEstimator


def main():
    """Demonstrate density estimation with VFA."""
    print("=" * 70)
    print("Vector Function Architecture - Density Estimation")
    print("=" * 70)

    # Initialize FHRR model
    key = jax.random.PRNGKey(42)
    model = create_fhrr_model(dim=2048, key=key)
    memory = VSAMemory(model)

    print("\n1. Generating Sample Data")
    print("-" * 70)

    # Generate mixture of Gaussians
    key1, key2, key3 = jax.random.split(key, 3)

    n_samples_per_mode = 50
    samples1 = jax.random.normal(key1, (n_samples_per_mode,)) * 0.5 + 2.0
    samples2 = jax.random.normal(key2, (n_samples_per_mode,)) * 0.8 + 6.0
    samples3 = jax.random.normal(key3, (n_samples_per_mode,)) * 0.3 + 9.0

    all_samples = jnp.concatenate([samples1, samples2, samples3])

    print(f"  Generated {len(all_samples)} samples from 3-mode Gaussian mixture")
    print(f"  Mode 1: mu=2.0, sigma=0.5 ({n_samples_per_mode} samples)")
    print(f"  Mode 2: mu=6.0, sigma=0.8 ({n_samples_per_mode} samples)")
    print(f"  Mode 3: mu=9.0, sigma=0.3 ({n_samples_per_mode} samples)")

    print("\n2. Fitting VFA Density Estimator")
    print("-" * 70)

    # Test different bandwidths
    bandwidths = [0.3, 0.5, 1.0]
    estimators = {}

    for bw in bandwidths:
        est = DensityEstimator(model, VSAMemory(model), bandwidth=bw)
        est.fit(all_samples)
        estimators[bw] = est
        print(f"  Fitted density estimator with bandwidth={bw}")

    print("\n3. Evaluating Densities")
    print("-" * 70)

    # Evaluate on grid
    x_grid = jnp.linspace(0, 12, 200)

    densities = {}
    for bw, est in estimators.items():
        density_values = est.evaluate(x_grid)
        densities[bw] = density_values
        peak_density = jnp.max(density_values)
        print(f"  Bandwidth {bw}: peak density = {peak_density:.4f}")

    print("\n4. Computing Ground Truth (for comparison)")
    print("-" * 70)

    # Compute true density (mixture of Gaussians)
    def gaussian(x, mu, sigma):
        return (
            1.0
            / (sigma * jnp.sqrt(2 * jnp.pi))
            * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)
        )

    true_density = (
        gaussian(x_grid, 2.0, 0.5) + gaussian(x_grid, 6.0, 0.8) + gaussian(x_grid, 9.0, 0.3)
    ) / 3.0

    print("  Computed true density from known mixture parameters")

    print("\n5. Visualization")
    print("-" * 70)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Different bandwidths
    ax = axes[0, 0]
    for bw in bandwidths:
        ax.plot(x_grid, densities[bw], linewidth=2, label=f"VFA (h={bw})")

    ax.plot(x_grid, true_density, "k--", linewidth=2, label="True density")
    ax.scatter(all_samples, jnp.zeros_like(all_samples), alpha=0.3, s=20, c="gray")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("VFA Density Estimation - Different Bandwidths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Focus on best bandwidth
    ax = axes[0, 1]
    best_bw = 0.5
    ax.plot(x_grid, densities[best_bw], linewidth=2.5, label=f"VFA (h={best_bw})")
    ax.plot(x_grid, true_density, "k--", linewidth=2, label="True density")
    ax.fill_between(x_grid, densities[best_bw], alpha=0.3)
    ax.scatter(all_samples, jnp.zeros_like(all_samples), alpha=0.5, s=30, c="red", marker="|")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"VFA Density Estimation (h={best_bw})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax = axes[1, 0]
    for bw in bandwidths:
        residuals = densities[bw] - true_density
        ax.plot(x_grid, residuals, linewidth=2, label=f"h={bw}")

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("Residual (VFA - True)")
    ax.set_title("Estimation Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    for bw in bandwidths:
        # Approximate CDF using cumulative sum
        dx = x_grid[1] - x_grid[0]
        cdf = jnp.cumsum(densities[bw]) * dx
        ax.plot(x_grid, cdf, linewidth=2, label=f"VFA (h={bw})")

    true_cdf = jnp.cumsum(true_density) * dx
    ax.plot(x_grid, true_cdf, "k--", linewidth=2, label="True CDF")
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Functions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vfa_density_estimation.png", dpi=150, bbox_inches="tight")
    print("  [OK] Plots saved to vfa_density_estimation.png")

    print("\n6. Quantitative Evaluation")
    print("-" * 70)

    # Compute errors
    for bw in bandwidths:
        mse = jnp.mean((densities[bw] - true_density) ** 2)
        mae = jnp.mean(jnp.abs(densities[bw] - true_density))
        print(f"  Bandwidth {bw}:")
        print(f"    MSE:  {mse:.6f}")
        print(f"    MAE:  {mae:.6f}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - VFA can estimate probability densities from samples")
    print("  - Bandwidth controls smoothness vs detail tradeoff")
    print("  - Works well for multimodal distributions")
    print("  - Compact hypervector representation of entire function")
    print("=" * 70)


if __name__ == "__main__":
    main()
