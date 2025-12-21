"""
Run all VSAX scientific experiments for journal paper.

This script executes both experiments sequentially:
1. Resonator convergence vs. composition depth and noise
2. Capacity-accuracy tradeoffs across VSA models

Total runtime: ~10-20 minutes depending on hardware.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.experiments.exp1_resonator_convergence import experiment_1
from examples.experiments.exp2_capacity_tradeoffs import experiment_2


def main():
    print("=" * 80)
    print("VSAX Scientific Experiments for Journal Publication")
    print("=" * 80)
    print()

    start_time = time.time()

    # Experiment 1
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Resonator Convergence")
    print("=" * 80)
    exp1_start = time.time()

    try:
        results1 = experiment_1(
            dimensions=[1024],
            depths=[1, 2, 3, 4],
            noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2],
            n_trials=20,
            output_dir="paper/experiments/exp1_results"
        )
        exp1_time = time.time() - exp1_start
        print(f"\nExperiment 1 completed in {exp1_time:.1f}s")
    except Exception as e:
        print(f"\nERROR in Experiment 1: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Capacity-Accuracy Tradeoffs")
    print("=" * 80)
    exp2_start = time.time()

    try:
        results2 = experiment_2(
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
        exp2_time = time.time() - exp2_start
        print(f"\nExperiment 2 completed in {exp2_time:.1f}s")
    except Exception as e:
        print(f"\nERROR in Experiment 2: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print("\nResults saved to:")
    print("  - paper/experiments/exp1_results/")
    print("  - paper/experiments/exp2_results/")
    print("\nPlots generated:")
    print("  - exp1_accuracy_vs_depth.pdf")
    print("  - exp1_accuracy_vs_noise.pdf")
    print("  - exp2_capacity_accuracy.pdf")
    print("  - exp2_capacity_similarity.pdf")


if __name__ == "__main__":
    main()
