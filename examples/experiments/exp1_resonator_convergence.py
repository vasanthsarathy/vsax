"""
Experiment 1: Operator Factorization vs. Composition Depth and Noise

Research Question: How does factorization performance degrade as a function of:
1. Operator composition depth (number of nested operators)
2. Noise level in the composite structure

Hypothesis: Recovery accuracy decreases with composition depth and noise,
but should maintain >80% accuracy for depth ≤3 with noise ≤0.1.

This experiment provides scientific insight into the factorization limits
of VSA operator compositions under structured compositional complexity.
"""

import jax
import jax.numpy as jnp
import numpy as np
from vsax import create_fhrr_model, VSAMemory
from vsax.operators import CliffordOperator
from vsax.similarity import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Tuple
import json


def create_compositional_structure(
    memory: VSAMemory,
    operators: List[CliffordOperator],
    concepts: List[str],
    depth: int,
    noise_std: float = 0.0,
    seed: int = 42
) -> Tuple[jnp.ndarray, List[str]]:
    """
    Create a compositional structure with nested operators.

    Args:
        memory: VSA memory with concepts
        operators: List of operators to compose
        concepts: List of concept names to bind
        depth: Number of operator compositions (1 = single op, 2 = op1∘op2, etc.)
        noise_std: Standard deviation of Gaussian noise to add
        seed: Random seed

    Returns:
        Composite vector and ground truth concepts used
    """
    key = jax.random.PRNGKey(seed)
    model = memory.model

    # Select concepts
    selected_concepts = concepts[:depth+1]

    # Start with first concept
    result = memory[selected_concepts[0]].vec

    # Apply nested operators: op_depth(...op_2(op_1(concept_1)) * concept_2...) * concept_depth
    for i in range(depth):
        # Apply operator to accumulated result
        op = operators[i % len(operators)]  # Cycle through operators if depth > len(operators)
        result = op.apply(model.rep_cls(result)).vec

        # Bind with next concept
        result = model.opset.bind(result, memory[selected_concepts[i+1]].vec)

    # Add noise
    if noise_std > 0:
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=result.shape) * noise_std
        # Add noise to real and imaginary parts separately
        result = result + noise.astype(result.dtype)
        # Renormalize
        result = result / jnp.linalg.norm(result)

    return result, selected_concepts


def run_direct_recovery(
    composite: jnp.ndarray,
    memory: VSAMemory,
    operators: List[CliffordOperator],
    depth: int
) -> Tuple[List[str], List[float]]:
    """
    Attempt to recover components from composite using direct unbinding.

    This iteratively unbinds operators and concepts in reverse order of composition.

    Returns:
        recovered_concepts: List of recovered concept names
        similarities: Similarity scores for each recovered concept
    """
    model = memory.model

    # For depth d, we need to unbind d concepts iteratively
    # The structure is: op_d(...op_2(op_1(c1) * c2) * ... * c_d+1)
    # We recover in reverse: c_d+1, c_d, ..., c_2, c_1
    recovered = []
    similarities = []

    current = composite

    # Unbind from the last concept to the first
    for d in range(depth, -1, -1):
        # Find the concept at this position by matching against memory
        best_match = None
        best_sim = -1.0

        for name in memory.keys():
            sim = float(cosine_similarity(current, memory[name].vec))
            if sim > best_sim:
                best_sim = sim
                best_match = name

        recovered.append(best_match)
        similarities.append(best_sim)

        # If not at the first concept, unbind this concept and invert the operator
        if d > 0:
            # Unbind the recovered concept
            current = model.opset.bind(current, model.opset.inverse(memory[best_match].vec))

            # Apply inverse operator to unwrap the next layer
            op = operators[(d-1) % len(operators)]
            current = op.inverse().apply(model.rep_cls(current)).vec

    return recovered, similarities


def experiment_1(
    dimensions: List[int] = [512, 1024, 2048],
    depths: List[int] = [1, 2, 3, 4, 5],
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
    n_trials: int = 10,
    output_dir: str = "results"
):
    """
    Main experiment: Test resonator convergence across depths and noise levels.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'dimensions': dimensions,
        'depths': depths,
        'noise_levels': noise_levels,
        'n_trials': n_trials,
        'accuracy': {},  # accuracy[dim][depth][noise] = [trial_accuracies]
        'similarity': {},  # similarity[dim][depth][noise] = [avg_similarities]
    }

    for dim in dimensions:
        print(f"\n=== Testing dimension {dim} ===")
        results['accuracy'][dim] = {}
        results['similarity'][dim] = {}

        # Create model and memory
        model = create_fhrr_model(dim=dim)
        memory = VSAMemory(model)

        # Add test concepts
        concepts = ['cat', 'dog', 'bird', 'fish', 'tree', 'car', 'house', 'book']
        for concept in concepts:
            memory.add(concept)

        # Create operators
        operators = [
            CliffordOperator.random(dim=dim, name=f"OP{i}", key=jax.random.PRNGKey(1000+i))
            for i in range(5)
        ]

        for depth in depths:
            print(f"  Depth {depth}...")
            results['accuracy'][dim][depth] = {}
            results['similarity'][dim][depth] = {}

            for noise in noise_levels:
                accuracies = []
                avg_sims = []

                for trial in range(n_trials):
                    # Create composite structure
                    composite, ground_truth = create_compositional_structure(
                        memory=memory,
                        operators=operators,
                        concepts=concepts,
                        depth=depth,
                        noise_std=noise,
                        seed=trial
                    )

                    # Attempt recovery
                    recovered, sims = run_direct_recovery(
                        composite=composite,
                        memory=memory,
                        operators=operators,
                        depth=depth
                    )

                    # Calculate accuracy (proportion of correctly recovered concepts)
                    correct = sum(1 for r, gt in zip(recovered, ground_truth) if r == gt)
                    accuracy = correct / len(ground_truth)
                    accuracies.append(accuracy)
                    avg_sims.append(np.mean(sims))

                results['accuracy'][dim][depth][noise] = accuracies
                results['similarity'][dim][depth][noise] = avg_sims

                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"    Noise {noise:.2f}: Accuracy = {mean_acc:.3f} ± {std_acc:.3f}")

    # Save results
    with open(f"{output_dir}/exp1_results.json", 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_results = {
            'dimensions': dimensions,
            'depths': depths,
            'noise_levels': noise_levels,
            'n_trials': n_trials,
            'accuracy': {
                str(dim): {
                    str(depth): {
                        str(noise): [float(x) for x in vals]
                        for noise, vals in noise_dict.items()
                    }
                    for depth, noise_dict in depth_dict.items()
                }
                for dim, depth_dict in results['accuracy'].items()
            },
            'similarity': {
                str(dim): {
                    str(depth): {
                        str(noise): [float(x) for x in vals]
                        for noise, vals in noise_dict.items()
                    }
                    for depth, noise_dict in depth_dict.items()
                }
                for dim, depth_dict in results['similarity'].items()
            }
        }
        json.dump(json_results, f, indent=2)

    # Generate plots
    plot_results(results, output_dir)

    return results


def plot_results(results, output_dir):
    """Generate publication-quality plots."""
    dimensions = results['dimensions']
    depths = results['depths']
    noise_levels = results['noise_levels']

    # Plot 1: Accuracy vs Depth (different noise levels)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(15, 4))
    if len(dimensions) == 1:
        axes = [axes]

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        for noise in noise_levels:
            means = []
            stds = []
            for depth in depths:
                accs = results['accuracy'][dim][depth][noise]
                means.append(np.mean(accs))
                stds.append(np.std(accs))

            ax.errorbar(depths, means, yerr=stds, marker='o', label=f'Noise={noise:.2f}', capsize=3)

        ax.set_xlabel('Composition Depth', fontsize=11)
        ax.set_ylabel('Recovery Accuracy', fontsize=11)
        ax.set_title(f'Dimension={dim}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_accuracy_vs_depth.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/exp1_accuracy_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Accuracy vs Noise (different depths)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(15, 4))
    if len(dimensions) == 1:
        axes = [axes]

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        for depth in depths[:4]:  # Only show first 4 depths for clarity
            means = []
            stds = []
            for noise in noise_levels:
                accs = results['accuracy'][dim][depth][noise]
                means.append(np.mean(accs))
                stds.append(np.std(accs))

            ax.errorbar(noise_levels, means, yerr=stds, marker='s', label=f'Depth={depth}', capsize=3)

        ax.set_xlabel('Noise Level (σ)', fontsize=11)
        ax.set_ylabel('Recovery Accuracy', fontsize=11)
        ax.set_title(f'Dimension={dim}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_accuracy_vs_noise.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/exp1_accuracy_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    print("Experiment 1: Resonator Convergence vs. Composition Depth and Noise")
    print("=" * 70)
    print("\nThis experiment tests the limits of resonator-based factorization")
    print("under increasing compositional complexity and noise.\n")

    # Run experiment with reasonable parameters
    results = experiment_1(
        dimensions=[1024],  # Focus on one dimension for speed
        depths=[1, 2, 3, 4],
        noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2],
        n_trials=20,
        output_dir="paper/experiments/exp1_results"
    )

    print("\nExperiment complete!")
    print(f"Results saved to paper/experiments/exp1_results/")
