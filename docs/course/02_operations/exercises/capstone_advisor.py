"""
Module 2 Capstone Project: VSA Model Selection Advisor

Build an intelligent advisor that:
1. Analyzes a task's requirements
2. Runs empirical tests with all three models
3. Provides a data-driven recommendation
4. Justifies the choice with evidence

This capstone integrates all Module 2 concepts:
- Deep understanding of FHRR, MAP, Binary operations
- Similarity metrics and search
- Benchmarking methodology
- Systematic model selection framework

Expected learning:
- Apply decision framework to real scenarios
- Empirical validation of theoretical trade-offs
- Build reusable model selection tools
- Understand when to trust theory vs measure empirically
"""

from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity
import jax.numpy as jnp
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TaskProfile:
    """
    Profile of a VSA task with requirements.
    """
    name: str
    num_concepts: int
    binding_depth: int
    needs_spatial: bool = False
    memory_limit_mb: float = float('inf')
    speed_critical: bool = False
    accuracy_threshold: float = 0.8


class ModelAdvisor:
    """
    Intelligent advisor for VSA model selection.

    Combines decision tree logic with empirical testing to recommend
    the best model for a given task.
    """

    def __init__(self, dim=2048):
        """
        Initialize the advisor.

        Args:
            dim: Dimension to use for testing
        """
        self.dim = dim
        self.test_results = {}

    def analyze_task(self, task: TaskProfile) -> Dict:
        """
        Analyze task requirements and run empirical tests.

        Args:
            task: Task profile with requirements

        Returns:
            Dictionary with analysis results and recommendation
        """
        print("=" * 80)
        print(f"Analyzing Task: {task.name}")
        print("=" * 80)

        print("\nTask Requirements:")
        print(f"  - Number of concepts: {task.num_concepts}")
        print(f"  - Binding depth: {task.binding_depth}")
        print(f"  - Spatial encoding needed: {task.needs_spatial}")
        print(f"  - Memory limit: {task.memory_limit_mb:.1f} MB")
        print(f"  - Speed critical: {task.speed_critical}")
        print(f"  - Accuracy threshold: {task.accuracy_threshold:.2f}")

        # Step 1: Apply decision tree
        print("\n" + "-" * 80)
        print("Step 1: Decision Tree Analysis")
        print("-" * 80)

        decision_tree_recommendation = self._apply_decision_tree(task)
        print(f"Decision tree recommends: {decision_tree_recommendation}")

        # Step 2: Check memory constraints
        print("\n" + "-" * 80)
        print("Step 2: Memory Constraint Check")
        print("-" * 80)

        memory_viable = self._check_memory_constraints(task)
        print(f"\nMemory viable models: {memory_viable}")

        # Step 3: Run empirical tests
        print("\n" + "-" * 80)
        print("Step 3: Empirical Performance Testing")
        print("-" * 80)

        test_results = self._run_empirical_tests(task, memory_viable)

        # Step 4: Generate recommendation
        print("\n" + "-" * 80)
        print("Step 4: Final Recommendation")
        print("-" * 80)

        recommendation = self._generate_recommendation(
            task, decision_tree_recommendation, memory_viable, test_results
        )

        return recommendation

    def _apply_decision_tree(self, task: TaskProfile) -> str:
        """
        Apply the decision tree from Lesson 2.4.
        """
        # Q1: Spatial encoding needed?
        if task.needs_spatial:
            print("  Q1: Spatial encoding needed? → YES")
            print("  → FHRR (only model supporting FPE/SSP)")
            return "FHRR"

        # Q2: Memory extremely limited?
        bytes_per_vec_binary = self.dim / 8
        total_mb_binary = (bytes_per_vec_binary * task.num_concepts) / 1024 / 1024

        if total_mb_binary * 32 > task.memory_limit_mb:  # Even Binary won't fit with 32x margin
            print("  Q2: Memory extremely limited? → YES")
            print("  → Binary (minimal memory footprint)")
            return "Binary"

        # Q3: Deep binding chains?
        if task.binding_depth > 3:
            print("  Q3: Deep binding chains (depth > 3)? → YES")
            print("  → FHRR (error accumulation in MAP)")
            return "FHRR"

        # Q4: Speed priority?
        if task.speed_critical:
            print("  Q4: Speed critical? → YES")
            print("  → MAP or Binary (faster than FHRR)")
            return "MAP"

        # Default
        print("  No strong constraint → Default to FHRR")
        return "FHRR"

    def _check_memory_constraints(self, task: TaskProfile) -> List[str]:
        """
        Check which models fit within memory constraints.
        """
        bytes_per_vec = {
            'FHRR': self.dim * 8,    # complex64
            'MAP': self.dim * 4,      # float32
            'Binary': self.dim / 8    # 1 bit
        }

        total_mb = {}
        viable = []

        for model, bytes_per in bytes_per_vec.items():
            total_bytes = bytes_per * task.num_concepts
            total_mb[model] = total_bytes / 1024 / 1024

            fits = total_mb[model] <= task.memory_limit_mb
            status = "✓ FITS" if fits else "✗ TOO LARGE"

            print(f"  {model:6s}: {total_mb[model]:>8.2f} MB  {status}")

            if fits:
                viable.append(model)

        return viable

    def _run_empirical_tests(self, task: TaskProfile, viable_models: List[str]) -> Dict:
        """
        Run empirical tests with each viable model.
        """
        results = {}

        for model_name in viable_models:
            print(f"\n  Testing {model_name}...")

            if model_name == 'FHRR':
                model = create_fhrr_model(dim=self.dim)
            elif model_name == 'MAP':
                model = create_map_model(dim=self.dim)
            else:  # Binary
                model = create_binary_model(dim=self.dim)

            memory = VSAMemory(model)

            # Test 1: Encoding speed
            concepts = [f"c_{i}" for i in range(min(task.num_concepts, 100))]
            start = time.time()
            memory.add_many(concepts)
            encoding_time = time.time() - start

            # Test 2: Binding speed
            a = memory["c_0"].vec
            b = memory["c_1"].vec

            # Warm-up
            for _ in range(10):
                model.opset.bind(a, b)

            num_ops = 1000
            start = time.time()
            for _ in range(num_ops):
                result = model.opset.bind(a, b)
                result.block_until_ready()
            binding_time = time.time() - start
            binding_speed = num_ops / binding_time

            # Test 3: Unbinding accuracy at task depth
            bound = a
            for _ in range(task.binding_depth):
                bound = model.opset.bind(bound, b)

            retrieved = bound
            for _ in range(task.binding_depth):
                if model_name == 'FHRR':
                    b_inv = jnp.conj(b)
                else:
                    b_inv = b
                retrieved = model.opset.bind(retrieved, b_inv)

            if model_name == 'Binary':
                accuracy = float(hamming_similarity(retrieved, a))
            else:
                accuracy = float(cosine_similarity(retrieved, a))

            results[model_name] = {
                'encoding_time': encoding_time,
                'binding_speed': binding_speed,
                'unbinding_accuracy': accuracy,
                'meets_accuracy': accuracy >= task.accuracy_threshold
            }

            print(f"    - Encoding time: {encoding_time:.4f}s")
            print(f"    - Binding speed: {binding_speed:.1f} ops/s")
            print(f"    - Unbinding accuracy (depth={task.binding_depth}): {accuracy:.6f}")
            print(f"    - Meets accuracy threshold ({task.accuracy_threshold}): {results[model_name]['meets_accuracy']}")

        self.test_results[task.name] = results
        return results

    def _generate_recommendation(
        self,
        task: TaskProfile,
        decision_tree_rec: str,
        memory_viable: List[str],
        empirical_results: Dict
    ) -> Dict:
        """
        Generate final recommendation with justification.
        """
        # Filter to models that meet accuracy threshold
        accurate_models = [
            model for model, results in empirical_results.items()
            if results['meets_accuracy']
        ]

        if not accurate_models:
            print("\n⚠️  WARNING: No models meet accuracy threshold!")
            print(f"   Consider increasing dimension or relaxing threshold.")
            # Fall back to best accuracy
            accurate_models = [
                max(empirical_results.keys(),
                    key=lambda m: empirical_results[m]['unbinding_accuracy'])
            ]

        # Among accurate models, pick fastest if speed is critical
        if task.speed_critical and len(accurate_models) > 1:
            recommended = max(
                accurate_models,
                key=lambda m: empirical_results[m]['binding_speed']
            )
            reason = f"Fastest among models meeting accuracy threshold (>{task.accuracy_threshold})"
        # Otherwise, prefer decision tree recommendation if it's viable and accurate
        elif decision_tree_rec in accurate_models:
            recommended = decision_tree_rec
            reason = "Decision tree recommendation confirmed by empirical testing"
        # Otherwise, pick highest accuracy
        else:
            recommended = max(
                accurate_models,
                key=lambda m: empirical_results[m]['unbinding_accuracy']
            )
            reason = "Highest unbinding accuracy"

        recommendation = {
            'task': task.name,
            'recommended_model': recommended,
            'reason': reason,
            'decision_tree_recommendation': decision_tree_rec,
            'memory_viable_models': memory_viable,
            'empirically_tested': list(empirical_results.keys()),
            'results': empirical_results[recommended],
            'all_results': empirical_results
        }

        # Print recommendation
        print(f"\n{'=' * 80}")
        print(f"RECOMMENDATION: {recommended}")
        print(f"{'=' * 80}")
        print(f"\nReason: {reason}")
        print(f"\nPerformance Metrics:")
        print(f"  - Encoding time: {empirical_results[recommended]['encoding_time']:.4f}s")
        print(f"  - Binding speed: {empirical_results[recommended]['binding_speed']:.1f} ops/s")
        print(f"  - Unbinding accuracy: {empirical_results[recommended]['unbinding_accuracy']:.6f}")

        print(f"\nAlternatives considered:")
        for model in empirical_results.keys():
            if model != recommended:
                print(f"  - {model}:")
                print(f"      Accuracy: {empirical_results[model]['unbinding_accuracy']:.6f}")
                print(f"      Speed: {empirical_results[model]['binding_speed']:.1f} ops/s")

        return recommendation


def demo_task_1_image_classification():
    """
    Task 1: Image classification (e.g., MNIST)
    """
    task = TaskProfile(
        name="Image Classification (MNIST)",
        num_concepts=1000,  # 10 classes + 784 pixel features + extras
        binding_depth=2,     # Pixel features → image vector
        needs_spatial=False,
        memory_limit_mb=1000,  # 1 GB available
        speed_critical=True,   # Real-time inference
        accuracy_threshold=0.7  # Classification uses similarity, approximate OK
    )

    advisor = ModelAdvisor(dim=2048)
    result = advisor.analyze_task(task)

    print("\n" + "=" * 80)
    print("Expected: MAP or Binary (speed priority, shallow binding)")
    print("=" * 80)

    return result


def demo_task_2_knowledge_graphs():
    """
    Task 2: Knowledge graph multi-hop reasoning
    """
    task = TaskProfile(
        name="Knowledge Graph Reasoning",
        num_concepts=5000,  # Large knowledge base
        binding_depth=4,     # Multi-hop queries
        needs_spatial=False,
        memory_limit_mb=5000,  # 5 GB available
        speed_critical=False,
        accuracy_threshold=0.95  # High accuracy needed for reasoning
    )

    advisor = ModelAdvisor(dim=4096)
    result = advisor.analyze_task(task)

    print("\n" + "=" * 80)
    print("Expected: FHRR (deep binding, high accuracy)")
    print("=" * 80)

    return result


def demo_task_3_embedded_device():
    """
    Task 3: Embedded device classification
    """
    task = TaskProfile(
        name="Embedded Device Classification",
        num_concepts=50,    # Small vocabulary
        binding_depth=1,    # Simple classification
        needs_spatial=False,
        memory_limit_mb=0.5,  # 512 KB limit (very constrained!)
        speed_critical=True,
        accuracy_threshold=0.7
    )

    advisor = ModelAdvisor(dim=2048)
    result = advisor.analyze_task(task)

    print("\n" + "=" * 80)
    print("Expected: Binary (severe memory constraint)")
    print("=" * 80)

    return result


def demo_task_4_spatial_robotics():
    """
    Task 4: Robot spatial memory
    """
    task = TaskProfile(
        name="Robot Spatial Memory",
        num_concepts=200,   # Landmarks and objects
        binding_depth=3,    # Location + object + properties
        needs_spatial=True,  # Must encode continuous coordinates
        memory_limit_mb=2000,
        speed_critical=False,
        accuracy_threshold=0.9
    )

    advisor = ModelAdvisor(dim=1024)
    result = advisor.analyze_task(task)

    print("\n" + "=" * 80)
    print("Expected: FHRR (spatial encoding requirement)")
    print("=" * 80)

    return result


def demo_task_5_document_hierarchy():
    """
    Task 5: Hierarchical document classification
    """
    task = TaskProfile(
        name="Hierarchical Document Classification",
        num_concepts=10000,  # Large corpus
        binding_depth=5,     # Category → Subcategory → Topic → Subtopic → Document
        needs_spatial=False,
        memory_limit_mb=10000,  # 10 GB available (GPU)
        speed_critical=False,
        accuracy_threshold=0.9
    )

    advisor = ModelAdvisor(dim=8192)  # Higher dimension for deep hierarchy
    result = advisor.analyze_task(task)

    print("\n" + "=" * 80)
    print("Expected: FHRR (very deep binding, high accuracy)")
    print("=" * 80)

    return result


def compare_all_tasks():
    """
    Compare recommendations across all demo tasks.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: All Task Recommendations")
    print("=" * 80)

    tasks = [
        ("Image Classification", demo_task_1_image_classification),
        ("Knowledge Graphs", demo_task_2_knowledge_graphs),
        ("Embedded Device", demo_task_3_embedded_device),
        ("Spatial Robotics", demo_task_4_spatial_robotics),
        ("Document Hierarchy", demo_task_5_document_hierarchy)
    ]

    results = []
    for name, task_fn in tasks:
        print(f"\n{'=' * 80}")
        print(f"Task {len(results) + 1}: {name}")
        print(f"{'=' * 80}")
        result = task_fn()
        results.append((name, result['recommended_model']))
        print(f"\n→ Recommended: {result['recommended_model']}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Task':<35s} {'Recommended Model':<20s}")
    print("-" * 60)
    for task_name, model in results:
        print(f"{task_name:<35s} {model:<20s}")


def main():
    """
    Run the VSA Model Selection Advisor capstone project.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 2 CAPSTONE PROJECT")
    print(" " * 15 + "VSA Model Selection Advisor")
    print("=" * 80)

    print("\nThis advisor combines:")
    print("  ✓ Decision tree logic from Lesson 2.4")
    print("  ✓ Empirical testing methodology")
    print("  ✓ Deep understanding of model trade-offs")
    print("  ✓ Systematic performance analysis")

    # Run all demo tasks
    compare_all_tasks()

    # Final summary
    print("\n" + "=" * 80)
    print("Capstone Project Complete!")
    print("=" * 80)
    print("\nWhat you learned:")
    print("  ✓ Apply decision framework to diverse tasks")
    print("  ✓ Validate theory with empirical measurements")
    print("  ✓ Balance multiple constraints (speed, accuracy, memory)")
    print("  ✓ Build reusable model selection tools")
    print("  ✓ Justify recommendations with data")

    print("\nKey Insights:")
    print("  1. FHRR: Best for deep binding, spatial encoding, high accuracy")
    print("  2. MAP: Best for speed, shallow binding, approximate OK")
    print("  3. Binary: Best for severe memory constraints, simple tasks")
    print("  4. Always validate: Theory guides, but measure for your task")

    print("\nCongratulations! You've mastered Module 2.")
    print("Ready for Module 3: Encoders & Applications")
    print("=" * 80)


if __name__ == "__main__":
    main()
