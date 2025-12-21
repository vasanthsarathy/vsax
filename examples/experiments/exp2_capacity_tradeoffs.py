"""
Experiment 2: Capacity-Accuracy Tradeoffs Across VSA Representations

Research Question: How do different VSA models (FHRR, MAP, Binary) trade off
capacity (number of items bundled) vs. retrieval accuracy under identical
encoding schemes?

Hypothesis: FHRR should maintain higher accuracy at larger capacities due to
complex-valued representations, while Binary should show earlier degradation
but with lower computational cost.

This experiment provides scientific insight into the fundamental capacity limits
of different VSA algebras under controlled conditions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import time


def bundle_and_query(
    memory: VSAMemory,
    n_items: int,
    n_queries: int = 100,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Bundle n_items and measure query accuracy.

    Args:
        memory: VSA memory with stored concepts
        n_items: Number of items to bundle
        n_queries: Number of random queries to test
        seed: Random seed

    Returns:
        mean_accuracy: Average retrieval accuracy
        mean_similarity: Average similarity to correct item
    """
    key = jax.random.PRNGKey(seed)
    model = memory.model

    # Get all concept names
    all_concepts = memory.keys()
    n_concepts = len(all_concepts)

    if n_items > n_concepts:
        raise ValueError(f"Cannot bundle {n_items} items with only {n_concepts} concepts")

    accuracies = []
    similarities = []

    for query_idx in range(n_queries):
        key, subkey = jax.random.split(key)

        # Randomly select n_items to bundle
        selected_indices = jax.random.choice(
            subkey,
            n_concepts,
            shape=(n_items,),
            replace=False
        )
        selected_concepts = [all_concepts[int(i)] for i in selected_indices]

        # Bundle the selected concepts
        vectors = [memory[name].vec for name in selected_concepts]
        bundled = model.opset.bundle(*vectors)

        # Query: try to retrieve a random item from the bundle
        key, subkey = jax.random.split(key)
        query_idx_local = int(jax.random.choice(subkey, n_items))
        target_concept = selected_concepts[query_idx_local]

        # Find best match in memory
        best_match = None
        best_sim = -float('inf')

        for name in all_concepts:
            if model.rep_cls.__name__ == 'BinaryHypervector':
                # Use Hamming similarity for binary
                sim = float(hamming_similarity(bundled, memory[name].vec))
            else:
                # Use cosine similarity for FHRR and MAP
                sim = float(cosine_similarity(bundled, memory[name].vec))

            if sim > best_sim:
                best_sim = sim
                best_match = name

        # Check if we retrieved the correct item
        is_correct = (best_match == target_concept)
        accuracies.append(1.0 if is_correct else 0.0)
        similarities.append(best_sim)

    return np.mean(accuracies), np.mean(similarities)


def experiment_2(
    dimensions: Dict[str, int] = None,
    capacities: List[int] = None,
    n_concepts: int = 200,
    n_queries_per_capacity: int = 50,
    n_trials: int = 5,
    output_dir: str = "results"
):
    """
    Main experiment: Test capacity-accuracy tradeoffs across VSA models.

    Args:
        dimensions: Dict mapping model name to dimension
        capacities: List of bundle sizes to test
        n_concepts: Number of concepts to create in memory
        n_queries_per_capacity: Number of queries per capacity test
        n_trials: Number of trials to average over
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if dimensions is None:
        dimensions = {
            'FHRR': 1024,
            'MAP': 1024,
            'Binary': 10000  # Binary typically needs higher dimensions
        }

    if capacities is None:
        capacities = [2, 5, 10, 20, 30, 50, 75, 100, 150]

    results = {
        'dimensions': dimensions,
        'capacities': capacities,
        'n_concepts': n_concepts,
        'n_queries_per_capacity': n_queries_per_capacity,
        'n_trials': n_trials,
        'accuracy': {},  # accuracy[model][capacity] = [trial_accuracies]
        'similarity': {},  # similarity[model][capacity] = [trial_similarities]
        'time': {}  # time[model][capacity] = [trial_times]
    }

    # Generate concept names
    concept_names = [f"concept_{i:03d}" for i in range(n_concepts)]

    for model_name, dim in dimensions.items():
        print(f"\n=== Testing {model_name} (dim={dim}) ===")
        results['accuracy'][model_name] = {}
        results['similarity'][model_name] = {}
        results['time'][model_name] = {}

        # Create model
        if model_name == 'FHRR':
            model = create_fhrr_model(dim=dim)
        elif model_name == 'MAP':
            model = create_map_model(dim=dim)
        elif model_name == 'Binary':
            model = create_binary_model(dim=dim)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Create memory and add concepts
        memory = VSAMemory(model)
        for name in concept_names:
            memory.add(name)

        for capacity in capacities:
            if capacity > n_concepts:
                print(f"  Skipping capacity {capacity} (exceeds {n_concepts} concepts)")
                continue

            print(f"  Capacity {capacity}...")
            trial_accs = []
            trial_sims = []
            trial_times = []

            for trial in range(n_trials):
                start_time = time.time()

                acc, sim = bundle_and_query(
                    memory=memory,
                    n_items=capacity,
                    n_queries=n_queries_per_capacity,
                    seed=trial
                )

                elapsed = time.time() - start_time

                trial_accs.append(acc)
                trial_sims.append(sim)
                trial_times.append(elapsed)

            results['accuracy'][model_name][capacity] = trial_accs
            results['similarity'][model_name][capacity] = trial_sims
            results['time'][model_name][capacity] = trial_times

            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            mean_time = np.mean(trial_times)
            print(f"    Accuracy: {mean_acc:.3f} ± {std_acc:.3f}, Time: {mean_time:.3f}s")

    # Save results
    with open(f"{output_dir}/exp2_results.json", 'w') as f:
        json_results = {
            'dimensions': dimensions,
            'capacities': capacities,
            'n_concepts': n_concepts,
            'n_queries_per_capacity': n_queries_per_capacity,
            'n_trials': n_trials,
            'accuracy': {
                model: {str(cap): [float(x) for x in vals] for cap, vals in cap_dict.items()}
                for model, cap_dict in results['accuracy'].items()
            },
            'similarity': {
                model: {str(cap): [float(x) for x in vals] for cap, vals in cap_dict.items()}
                for model, cap_dict in results['similarity'].items()
            },
            'time': {
                model: {str(cap): [float(x) for x in vals] for cap, vals in cap_dict.items()}
                for model, cap_dict in results['time'].items()
            }
        }
        json.dump(json_results, f, indent=2)

    # Generate plots
    plot_results(results, output_dir)

    return results


def plot_results(results, output_dir):
    """Generate publication-quality plots."""
    capacities = results['capacities']
    models = list(results['accuracy'].keys())

    # Plot 1: Accuracy vs Capacity
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {'FHRR': '#2E86AB', 'MAP': '#A23B72', 'Binary': '#F18F01'}
    markers = {'FHRR': 'o', 'MAP': 's', 'Binary': '^'}

    for model in models:
        caps = []
        means = []
        stds = []

        for cap in capacities:
            if cap in results['accuracy'][model]:
                accs = results['accuracy'][model][cap]
                caps.append(cap)
                means.append(np.mean(accs))
                stds.append(np.std(accs))

        ax.errorbar(
            caps, means, yerr=stds,
            marker=markers[model],
            label=model,
            color=colors[model],
            capsize=4,
            linewidth=2,
            markersize=6
        )

    ax.set_xlabel('Bundle Capacity (number of items)', fontsize=12)
    ax.set_ylabel('Retrieval Accuracy', fontsize=12)
    ax.set_title('VSA Capacity-Accuracy Tradeoffs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, max(capacities) + 5])

    # Add horizontal line at chance level (1/n_concepts)
    chance = 1.0 / results['n_concepts']
    ax.axhline(y=chance, color='gray', linestyle=':', alpha=0.5, label=f'Chance ({chance:.3f})')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_capacity_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/exp2_capacity_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Similarity vs Capacity
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model in models:
        caps = []
        means = []
        stds = []

        for cap in capacities:
            if cap in results['similarity'][model]:
                sims = results['similarity'][model][cap]
                caps.append(cap)
                means.append(np.mean(sims))
                stds.append(np.std(sims))

        ax.errorbar(
            caps, means, yerr=stds,
            marker=markers[model],
            label=model,
            color=colors[model],
            capsize=4,
            linewidth=2,
            markersize=6
        )

    ax.set_xlabel('Bundle Capacity (number of items)', fontsize=12)
    ax.set_ylabel('Mean Similarity to Target', fontsize=12)
    ax.set_title('Similarity Degradation with Capacity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_capacity_similarity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/exp2_capacity_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    print("Experiment 2: Capacity-Accuracy Tradeoffs Across VSA Models")
    print("=" * 70)
    print("\nThis experiment compares fundamental capacity limits across")
    print("different VSA representations under controlled conditions.\n")

    # Run experiment
    results = experiment_2(
        dimensions={
            'FHRR': 1024,
            'MAP': 1024,
            'Binary': 10000
        },
        capacities=[2, 5, 10, 15, 20, 30, 40, 50, 75, 100],
        n_concepts=200,
        n_queries_per_capacity=50,
        n_trials=10,
        output_dir="paper/experiments/exp2_results"
    )

    print("\nExperiment complete!")
    print(f"Results saved to paper/experiments/exp2_results/")
