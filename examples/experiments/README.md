# VSAX Scientific Experiments

This directory contains hypothesis-driven experiments for the VSAX journal paper. These experiments provide scientific insights into VSA behavior beyond performance benchmarks.

## Experiments

### Experiment 1: Resonator Convergence vs. Composition Depth and Noise

**Research Question**: How does resonator network performance degrade with increasing compositional complexity and noise?

**Hypothesis**: Resonator accuracy decreases with composition depth and noise, but should maintain >80% accuracy for depth ≤3 with noise ≤0.1.

**Variables**:
- **Independent**: Composition depth (1-5 nested operators), Noise level (0.0-0.2)
- **Dependent**: Recovery accuracy, convergence iterations
- **Controlled**: Dimension (1024), number of concepts (8), number of trials (20)

**Scientific Insight**: Reveals fundamental limits of VSA factorization under structured compositional complexity.

**Run**:
```bash
uv run python examples/experiments/exp1_resonator_convergence.py
```

**Outputs**:
- `paper/experiments/exp1_results/exp1_results.json` - Raw numerical results
- `paper/experiments/exp1_results/exp1_accuracy_vs_depth.pdf` - Accuracy vs depth plot
- `paper/experiments/exp1_results/exp1_accuracy_vs_noise.pdf` - Accuracy vs noise plot

### Experiment 2: Capacity-Accuracy Tradeoffs Across VSA Models

**Research Question**: How do different VSA representations (FHRR, MAP, Binary) trade off capacity vs. accuracy?

**Hypothesis**: FHRR maintains higher accuracy at larger capacities due to complex representations, while Binary shows earlier degradation.

**Variables**:
- **Independent**: Bundle capacity (2-100 items), VSA model type
- **Dependent**: Retrieval accuracy, similarity to target
- **Controlled**: Number of concepts (200), queries per capacity (50), trials (10)

**Scientific Insight**: Quantifies fundamental capacity limits of different VSA algebras under controlled conditions.

**Run**:
```bash
uv run python examples/experiments/exp2_capacity_tradeoffs.py
```

**Outputs**:
- `paper/experiments/exp2_results/exp2_results.json` - Raw numerical results
- `paper/experiments/exp2_results/exp2_capacity_accuracy.pdf` - Capacity-accuracy curves
- `paper/experiments/exp2_results/exp2_capacity_similarity.pdf` - Similarity degradation curves

## Running All Experiments

```bash
uv run python examples/experiments/run_all.py
```

This will execute both experiments sequentially and generate all results and plots.

## Interpreting Results

### Experiment 1 Interpretation

- **Accuracy vs Depth**: Shows how compositional complexity affects factorization
- **Accuracy vs Noise**: Reveals robustness to perturbations
- **Key Findings**: Expected to show exponential degradation with depth, linear degradation with noise

### Experiment 2 Interpretation

- **Capacity Curves**: Shows where each VSA model "breaks down"
- **Model Comparison**: Quantifies tradeoffs between representation types
- **Key Findings**: Expected to show FHRR > MAP > Binary in capacity, with crossover points

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{sarathy2025vsax,
  title={VSAX: A GPU-Accelerated Vector Symbolic Algebra Library for JAX},
  author={Sarathy, Vasanth},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Requirements

- VSAX >= 1.1.0
- JAX >= 0.4.23
- NumPy >= 1.24.3
- Matplotlib >= 3.7.0

All experiments use fixed random seeds for reproducibility.
