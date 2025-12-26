"""
Module 2 Exercise 2: Comprehensive Model Benchmarking

This exercise benchmarks FHRR, MAP, and Binary models across multiple dimensions:
- Encoding speed
- Binding operation speed
- Bundling operation speed
- Unbinding accuracy at different depths
- Memory footprint
- Similarity computation speed

Tasks:
1. Benchmark encoding performance (creating basis vectors)
2. Benchmark binding and bundling speeds
3. Measure unbinding accuracy vs binding depth
4. Compare memory footprints
5. Test batch operation performance
6. Generate comparison plots

Expected learning:
- Empirical understanding of model trade-offs
- Performance characteristics of each model
- When speed vs accuracy matters
- Memory vs compute trade-offs
"""

import jax.numpy as jnp
import jax.random as random
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity
import time
import matplotlib.pyplot as plt


class ModelBenchmark:
    """
    Comprehensive benchmarking suite for VSA models.
    """

    def __init__(self, dim=2048, num_concepts=100):
        """
        Initialize benchmark suite.

        Args:
            dim: Vector dimensionality
            num_concepts: Number of concepts to encode
        """
        self.dim = dim
        self.num_concepts = num_concepts
        self.results = {}

    def benchmark_encoding(self):
        """
        Benchmark encoding speed for all three models.
        """
        print("=" * 60)
        print("Benchmark 1: Encoding Speed")
        print("=" * 60)

        concepts = [f"concept_{i}" for i in range(self.num_concepts)]

        # FHRR
        start = time.time()
        fhrr_model = create_fhrr_model(dim=self.dim)
        fhrr_mem = VSAMemory(fhrr_model)
        fhrr_mem.add_many(concepts)
        fhrr_time = time.time() - start
        fhrr_speed = self.num_concepts / fhrr_time

        # MAP
        start = time.time()
        map_model = create_map_model(dim=self.dim)
        map_mem = VSAMemory(map_model)
        map_mem.add_many(concepts)
        map_time = time.time() - start
        map_speed = self.num_concepts / map_time

        # Binary
        start = time.time()
        binary_model = create_binary_model(dim=self.dim)
        binary_mem = VSAMemory(binary_model)
        binary_mem.add_many(concepts)
        binary_time = time.time() - start
        binary_speed = self.num_concepts / binary_time

        print(f"\nEncoding {self.num_concepts} concepts (d={self.dim}):")
        print(f"  FHRR:   {fhrr_time:.4f}s ({fhrr_speed:.1f} concepts/s)")
        print(f"  MAP:    {map_time:.4f}s ({map_speed:.1f} concepts/s)")
        print(f"  Binary: {binary_time:.4f}s ({binary_speed:.1f} concepts/s)")

        fastest = min(fhrr_speed, map_speed, binary_speed)
        print(f"\nSpeedup vs slowest:")
        print(f"  FHRR:   {fhrr_speed / fastest:.2f}x")
        print(f"  MAP:    {map_speed / fastest:.2f}x")
        print(f"  Binary: {binary_speed / fastest:.2f}x")

        self.results['encoding'] = {
            'fhrr': fhrr_time,
            'map': map_time,
            'binary': binary_time
        }

        return fhrr_mem, map_mem, binary_mem

    def benchmark_binding(self, fhrr_mem, map_mem, binary_mem, num_ops=1000):
        """
        Benchmark binding operation speed.
        """
        print("\n" + "=" * 60)
        print("Benchmark 2: Binding Operation Speed")
        print("=" * 60)

        # Get two vectors for binding
        fhrr_a = fhrr_mem["concept_0"].vec
        fhrr_b = fhrr_mem["concept_1"].vec

        map_a = map_mem["concept_0"].vec
        map_b = map_mem["concept_1"].vec

        binary_a = binary_mem["concept_0"].vec
        binary_b = binary_mem["concept_1"].vec

        # Warm-up (JIT compilation)
        for _ in range(10):
            fhrr_mem.model.opset.bind(fhrr_a, fhrr_b)
            map_mem.model.opset.bind(map_a, map_b)
            binary_mem.model.opset.bind(binary_a, binary_b)

        # FHRR binding
        start = time.time()
        for _ in range(num_ops):
            result = fhrr_mem.model.opset.bind(fhrr_a, fhrr_b)
            result.block_until_ready()  # Ensure computation completes
        fhrr_time = time.time() - start
        fhrr_speed = num_ops / fhrr_time

        # MAP binding
        start = time.time()
        for _ in range(num_ops):
            result = map_mem.model.opset.bind(map_a, map_b)
            result.block_until_ready()
        map_time = time.time() - start
        map_speed = num_ops / map_time

        # Binary binding
        start = time.time()
        for _ in range(num_ops):
            result = binary_mem.model.opset.bind(binary_a, binary_b)
            result.block_until_ready()
        binary_time = time.time() - start
        binary_speed = num_ops / binary_time

        print(f"\nBinding operations ({num_ops} operations):")
        print(f"  FHRR:   {fhrr_time:.4f}s ({fhrr_speed:.1f} ops/s)")
        print(f"  MAP:    {map_time:.4f}s ({map_speed:.1f} ops/s)")
        print(f"  Binary: {binary_time:.4f}s ({binary_speed:.1f} ops/s)")

        fastest = max(fhrr_speed, map_speed, binary_speed)
        print(f"\nSpeed relative to fastest:")
        print(f"  FHRR:   {fhrr_speed / fastest:.2f}x")
        print(f"  MAP:    {map_speed / fastest:.2f}x")
        print(f"  Binary: {binary_speed / fastest:.2f}x")

        self.results['binding'] = {
            'fhrr': fhrr_speed,
            'map': map_speed,
            'binary': binary_speed
        }

    def benchmark_bundling(self, fhrr_mem, map_mem, binary_mem, bundle_size=20):
        """
        Benchmark bundling operation speed.
        """
        print("\n" + "=" * 60)
        print("Benchmark 3: Bundling Operation Speed")
        print("=" * 60)

        # Get vectors for bundling
        fhrr_vecs = [fhrr_mem[f"concept_{i}"].vec for i in range(bundle_size)]
        map_vecs = [map_mem[f"concept_{i}"].vec for i in range(bundle_size)]
        binary_vecs = [binary_mem[f"concept_{i}"].vec for i in range(bundle_size)]

        num_ops = 1000

        # Warm-up
        for _ in range(10):
            fhrr_mem.model.opset.bundle(*fhrr_vecs)
            map_mem.model.opset.bundle(*map_vecs)
            binary_mem.model.opset.bundle(*binary_vecs)

        # FHRR bundling
        start = time.time()
        for _ in range(num_ops):
            result = fhrr_mem.model.opset.bundle(*fhrr_vecs)
            result.block_until_ready()
        fhrr_time = time.time() - start
        fhrr_speed = num_ops / fhrr_time

        # MAP bundling
        start = time.time()
        for _ in range(num_ops):
            result = map_mem.model.opset.bundle(*map_vecs)
            result.block_until_ready()
        map_time = time.time() - start
        map_speed = num_ops / map_time

        # Binary bundling
        start = time.time()
        for _ in range(num_ops):
            result = binary_mem.model.opset.bundle(*binary_vecs)
            result.block_until_ready()
        binary_time = time.time() - start
        binary_speed = num_ops / binary_time

        print(f"\nBundling {bundle_size} vectors ({num_ops} operations):")
        print(f"  FHRR:   {fhrr_time:.4f}s ({fhrr_speed:.1f} ops/s)")
        print(f"  MAP:    {map_time:.4f}s ({map_speed:.1f} ops/s)")
        print(f"  Binary: {binary_time:.4f}s ({binary_speed:.1f} ops/s)")

        fastest = max(fhrr_speed, map_speed, binary_speed)
        print(f"\nSpeed relative to fastest:")
        print(f"  FHRR:   {fhrr_speed / fastest:.2f}x")
        print(f"  MAP:    {map_speed / fastest:.2f}x")
        print(f"  Binary: {binary_speed / fastest:.2f}x")

        self.results['bundling'] = {
            'fhrr': fhrr_speed,
            'map': map_speed,
            'binary': binary_speed
        }

    def benchmark_unbinding_accuracy(self, max_depth=7):
        """
        Benchmark unbinding accuracy across different binding depths.
        """
        print("\n" + "=" * 60)
        print("Benchmark 4: Unbinding Accuracy vs Depth")
        print("=" * 60)

        # Create fresh models
        fhrr_model = create_fhrr_model(dim=self.dim)
        map_model = create_map_model(dim=self.dim)
        binary_model = create_binary_model(dim=self.dim)

        fhrr_mem = VSAMemory(fhrr_model)
        map_mem = VSAMemory(map_model)
        binary_mem = VSAMemory(binary_model)

        fhrr_mem.add_many(["a", "b"])
        map_mem.add_many(["a", "b"])
        binary_mem.add_many(["a", "b"])

        print(f"\nUnbinding accuracy (d={self.dim}):")
        print(f"{'Depth':>6s}  {'FHRR':>10s}  {'MAP':>10s}  {'Binary':>10s}")
        print("-" * 50)

        fhrr_results = []
        map_results = []
        binary_results = []

        for depth in range(1, max_depth + 1):
            # FHRR
            fhrr_bound = fhrr_mem["a"].vec
            for _ in range(depth):
                fhrr_bound = fhrr_model.opset.bind(fhrr_bound, fhrr_mem["b"].vec)

            fhrr_retrieved = fhrr_bound
            for _ in range(depth):
                fhrr_retrieved = fhrr_model.opset.bind(fhrr_retrieved, jnp.conj(fhrr_mem["b"].vec))

            fhrr_sim = float(cosine_similarity(fhrr_retrieved, fhrr_mem["a"].vec))
            fhrr_results.append((depth, fhrr_sim))

            # MAP
            map_bound = map_mem["a"].vec
            for _ in range(depth):
                map_bound = map_model.opset.bind(map_bound, map_mem["b"].vec)

            map_retrieved = map_bound
            for _ in range(depth):
                map_retrieved = map_model.opset.bind(map_retrieved, map_mem["b"].vec)

            map_sim = float(cosine_similarity(map_retrieved, map_mem["a"].vec))
            map_results.append((depth, map_sim))

            # Binary
            binary_bound = binary_mem["a"].vec
            for _ in range(depth):
                binary_bound = binary_model.opset.bind(binary_bound, binary_mem["b"].vec)

            binary_retrieved = binary_bound
            for _ in range(depth):
                binary_retrieved = binary_model.opset.bind(binary_retrieved, binary_mem["b"].vec)

            binary_sim = float(hamming_similarity(binary_retrieved, binary_mem["a"].vec))
            binary_results.append((depth, binary_sim))

            print(f"{depth:6d}  {fhrr_sim:10.6f}  {map_sim:10.6f}  {binary_sim:10.6f}")

        self.results['unbinding_accuracy'] = {
            'fhrr': fhrr_results,
            'map': map_results,
            'binary': binary_results
        }

        return fhrr_results, map_results, binary_results

    def benchmark_memory_footprint(self):
        """
        Compare memory footprints of the three models.
        """
        print("\n" + "=" * 60)
        print("Benchmark 5: Memory Footprint")
        print("=" * 60)

        # Bytes per element
        fhrr_bytes = 8  # complex64 = 2 × float32 = 2 × 4 = 8 bytes
        map_bytes = 4  # float32 = 4 bytes
        binary_bytes = 1 / 8  # 1 bit = 1/8 byte

        # Bytes per vector
        fhrr_vec = self.dim * fhrr_bytes
        map_vec = self.dim * map_bytes
        binary_vec = self.dim * binary_bytes

        # Bytes for entire memory (num_concepts vectors)
        fhrr_total = fhrr_vec * self.num_concepts
        map_total = map_vec * self.num_concepts
        binary_total = binary_vec * self.num_concepts

        print(f"\nMemory per vector (d={self.dim}):")
        print(f"  FHRR:   {fhrr_vec:>10.1f} bytes ({fhrr_vec/1024:.2f} KB)")
        print(f"  MAP:    {map_vec:>10.1f} bytes ({map_vec/1024:.2f} KB)")
        print(f"  Binary: {binary_vec:>10.1f} bytes ({binary_vec/1024:.2f} KB)")

        print(f"\nMemory for {self.num_concepts} vectors:")
        print(f"  FHRR:   {fhrr_total:>10.1f} bytes ({fhrr_total/1024:.2f} KB, {fhrr_total/1024/1024:.2f} MB)")
        print(f"  MAP:    {map_total:>10.1f} bytes ({map_total/1024:.2f} KB, {map_total/1024/1024:.2f} MB)")
        print(f"  Binary: {binary_total:>10.1f} bytes ({binary_total/1024:.2f} KB, {binary_total/1024/1024:.2f} MB)")

        print(f"\nMemory ratio (vs Binary):")
        print(f"  FHRR:   {fhrr_total / binary_total:.1f}x")
        print(f"  MAP:    {map_total / binary_total:.1f}x")
        print(f"  Binary: 1.0x (baseline)")

        self.results['memory'] = {
            'fhrr': fhrr_total,
            'map': map_total,
            'binary': binary_total
        }

    def plot_results(self):
        """
        Visualize benchmark results.
        """
        print("\n" + "=" * 60)
        print("Generating Comparison Plots")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Unbinding Accuracy vs Depth
        ax1 = axes[0, 0]
        fhrr_data = self.results['unbinding_accuracy']['fhrr']
        map_data = self.results['unbinding_accuracy']['map']
        binary_data = self.results['unbinding_accuracy']['binary']

        depths_fhrr, sims_fhrr = zip(*fhrr_data)
        depths_map, sims_map = zip(*map_data)
        depths_binary, sims_binary = zip(*binary_data)

        ax1.plot(depths_fhrr, sims_fhrr, 'o-', label='FHRR', linewidth=2, markersize=8)
        ax1.plot(depths_map, sims_map, 's-', label='MAP', linewidth=2, markersize=8)
        ax1.plot(depths_binary, sims_binary, '^-', label='Binary', linewidth=2, markersize=8)
        ax1.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
        ax1.set_xlabel('Binding Depth', fontsize=11)
        ax1.set_ylabel('Unbinding Similarity', fontsize=11)
        ax1.set_title(f'Unbinding Accuracy vs Depth (d={self.dim})', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.4, 1.05])

        # Plot 2: Operation Speed Comparison
        ax2 = axes[0, 1]
        operations = ['Binding', 'Bundling']
        fhrr_speeds = [self.results['binding']['fhrr'], self.results['bundling']['fhrr']]
        map_speeds = [self.results['binding']['map'], self.results['bundling']['map']]
        binary_speeds = [self.results['binding']['binary'], self.results['bundling']['binary']]

        x = jnp.arange(len(operations))
        width = 0.25

        ax2.bar(x - width, fhrr_speeds, width, label='FHRR', alpha=0.8)
        ax2.bar(x, map_speeds, width, label='MAP', alpha=0.8)
        ax2.bar(x + width, binary_speeds, width, label='Binary', alpha=0.8)
        ax2.set_xlabel('Operation Type', fontsize=11)
        ax2.set_ylabel('Operations per Second', fontsize=11)
        ax2.set_title('Operation Speed Comparison', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(operations)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Memory Footprint
        ax3 = axes[1, 0]
        models = ['FHRR', 'MAP', 'Binary']
        memory_mb = [
            self.results['memory']['fhrr'] / 1024 / 1024,
            self.results['memory']['map'] / 1024 / 1024,
            self.results['memory']['binary'] / 1024 / 1024
        ]

        bars = ax3.bar(models, memory_mb, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_ylabel('Memory (MB)', fontsize=11)
        ax3.set_title(f'Memory Footprint ({self.num_concepts} vectors, d={self.dim})', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

        # Plot 4: Trade-off Matrix (Speed vs Accuracy)
        ax4 = axes[1, 1]

        # Use depth=3 accuracy and binding speed
        depth_3_sims = {
            'FHRR': [sim for d, sim in fhrr_data if d == 3][0],
            'MAP': [sim for d, sim in map_data if d == 3][0],
            'Binary': [sim for d, sim in binary_data if d == 3][0]
        }

        speeds = [self.results['binding']['fhrr'],
                 self.results['binding']['map'],
                 self.results['binding']['binary']]
        accuracies = [depth_3_sims['FHRR'], depth_3_sims['MAP'], depth_3_sims['Binary']]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, model in enumerate(models):
            ax4.scatter(speeds[i], accuracies[i], s=300, alpha=0.6,
                       color=colors[i], label=model, edgecolors='black', linewidth=2)
            ax4.annotate(model, (speeds[i], accuracies[i]),
                        fontsize=12, ha='center', va='center', weight='bold')

        ax4.set_xlabel('Binding Speed (ops/s)', fontsize=11)
        ax4.set_ylabel('Unbinding Accuracy (depth=3)', fontsize=11)
        ax4.set_title('Speed vs Accuracy Trade-off', fontsize=12)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_benchmark_results.png', dpi=150)
        print("\nSaved: model_benchmark_results.png")
        plt.show()


def main():
    """
    Run comprehensive model benchmarks.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 2 EXERCISE 2")
    print(" " * 15 + "Comprehensive Model Benchmarking")
    print("=" * 80)

    # Run benchmarks
    benchmark = ModelBenchmark(dim=2048, num_concepts=100)

    fhrr_mem, map_mem, binary_mem = benchmark.benchmark_encoding()
    benchmark.benchmark_binding(fhrr_mem, map_mem, binary_mem)
    benchmark.benchmark_bundling(fhrr_mem, map_mem, binary_mem)
    benchmark.benchmark_unbinding_accuracy(max_depth=7)
    benchmark.benchmark_memory_footprint()
    benchmark.plot_results()

    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print("\nFHRR:")
    print("  + Exact unbinding (>0.99 similarity at all depths)")
    print("  + Supports spatial encoding (FPE, SSP)")
    print("  - Slowest operations (FFT overhead)")
    print("  - Highest memory (8 bytes/element)")

    print("\nMAP:")
    print("  + Fastest operations (element-wise multiply)")
    print("  + Moderate memory (4 bytes/element)")
    print("  - Approximate unbinding (~0.7-0.8 similarity)")
    print("  - Error accumulates with depth")

    print("\nBinary:")
    print("  + Minimal memory (1 bit/element)")
    print("  + Fast operations (XOR, majority vote)")
    print("  + Self-inverse property")
    print("  - No spatial encoding support")
    print("  - Binary representation limits expressiveness")

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
