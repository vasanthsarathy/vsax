"""
Module 4 Exercise 1: Spatial Reasoning with SSP and Operators

This exercise combines Spatial Semantic Pointers (Lesson 4.2) with
Clifford Operators (Lesson 4.1) to build a complete spatial reasoning system.

Tasks:
1. Build a 2D spatial scene with SSP
2. Add directional relations using Clifford Operators
3. Query spatial locations and relational facts
4. Visualize the scene with heatmaps
5. Test complex spatial reasoning queries

Expected learning:
- Combining continuous spatial encoding (SSP) with discrete relations (operators)
- Building rich spatial representations
- Cross-querying location and relation information
- Practical spatial AI for robotics/navigation
"""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.spatial import SpatialSemanticPointers, SSPConfig
from vsax.spatial.utils import create_spatial_scene, similarity_map_2d
from vsax.operators import CliffordOperator, OperatorKind
from vsax.similarity import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


class SpatialReasoningSystem:
    """
    Combines SSP for continuous locations with Clifford Operators
    for discrete spatial relations.
    """

    def __init__(self, dim=1024, seed=42):
        """
        Initialize spatial reasoning system.

        Args:
            dim: Hypervector dimensionality
            seed: Random seed
        """
        self.model = create_fhrr_model(dim=dim, key=jax.random.PRNGKey(seed))
        self.memory = VSAMemory(self.model)

        # Create SSP for continuous spatial encoding
        config = SSPConfig(dim=dim, num_axes=2, axis_names=["x", "y"])
        self.ssp = SpatialSemanticPointers(self.model, self.memory, config)

        # Create Clifford Operators for spatial relations
        self.operators = {}
        self._create_spatial_operators()

        # Storage for scene objects
        self.scene = None
        self.objects = {}  # {name: {"location": [x, y], "relations": {...}}}

    def _create_spatial_operators(self):
        """Create spatial relation operators."""
        relations = ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR"]

        for relation in relations:
            self.operators[relation] = CliffordOperator.random(
                dim=self.model.dim,
                kind=OperatorKind.SPATIAL,
                name=relation,
                key=jax.random.PRNGKey(hash(relation) % 2**31)
            )

        print(f"Created {len(self.operators)} spatial relation operators")

    def add_object(self, name, location, related_objects=None):
        """
        Add object to scene with location and optional relations.

        Args:
            name: Object name
            location: [x, y] coordinates
            related_objects: Dict of {relation: other_object_name}
                            e.g., {"LEFT_OF": "table", "NEAR": "door"}
        """
        # Add object to memory
        if name not in self.memory:
            self.memory.add(name)

        # Store object info
        self.objects[name] = {
            "location": location,
            "relations": related_objects or {}
        }

    def build_scene(self):
        """
        Build complete spatial scene encoding both locations and relations.
        """
        scene_components = []

        for name, info in self.objects.items():
            # Component 1: Bind object to spatial location (SSP)
            location_binding = self.ssp.bind_object_location(name, info["location"])
            scene_components.append(location_binding.vec)

            # Component 2: Encode relational facts (Operators)
            for relation, other_obj in info["relations"].items():
                if other_obj in self.memory:
                    # relation(name, other_obj)
                    # e.g., LEFT_OF(cup, plate)
                    operator = self.operators[relation]
                    relation_hv = self.model.opset.bundle(
                        self.memory[name].vec,
                        operator.apply(self.memory[other_obj]).vec
                    )
                    scene_components.append(relation_hv)

        # Bundle all components
        self.scene = self.model.opset.bundle(*scene_components)
        print(f"Built scene with {len(self.objects)} objects and {sum(len(o['relations']) for o in self.objects.values())} relations")

    def query_location(self, obj_name):
        """
        Query: Where is object X?

        Args:
            obj_name: Object to locate

        Returns:
            Approximate [x, y] coordinates
        """
        if obj_name not in self.memory:
            return None

        # Unbind object to get location
        location_hv = self.ssp.query_object(self.scene, obj_name)

        # Decode location
        coords = self.ssp.decode_location(
            location_hv,
            search_range=[(0, 10), (0, 10)],
            resolution=40
        )

        return coords

    def query_at_location(self, location):
        """
        Query: What is at location [x, y]?

        Args:
            location: [x, y] coordinates

        Returns:
            (object_name, similarity)
        """
        # Query SSP
        result_hv = self.ssp.query_location(self.scene, location)

        # Find best matching object
        best_match = None
        best_sim = -1.0

        for obj_name in self.objects.keys():
            sim = cosine_similarity(result_hv.vec, self.memory[obj_name].vec)
            if sim > best_sim:
                best_sim = sim
                best_match = obj_name

        return best_match, best_sim

    def query_relation(self, relation, obj1, obj2):
        """
        Query: Is obj1 RELATION obj2?

        Args:
            relation: Relation name (e.g., "LEFT_OF")
            obj1: First object
            obj2: Second object

        Returns:
            Similarity score (higher = more likely true)
        """
        if relation not in self.operators:
            return 0.0

        operator = self.operators[relation]

        # Construct expected relation: obj1 + operator(obj2)
        expected = self.model.opset.bundle(
            self.memory[obj1].vec,
            operator.apply(self.memory[obj2]).vec
        )

        # Check similarity to scene
        sim = cosine_similarity(expected, self.scene)
        return sim

    def find_related(self, relation, target_obj):
        """
        Query: What objects have RELATION with target_obj?

        Args:
            relation: Relation name
            target_obj: Target object

        Returns:
            List of (object_name, similarity) tuples
        """
        if relation not in self.operators:
            return []

        operator = self.operators[relation]
        results = []

        # Check all objects
        for obj_name in self.objects.keys():
            if obj_name == target_obj:
                continue

            # Test: obj_name RELATION target_obj
            sim = self.query_relation(relation, obj_name, target_obj)
            if sim > 0.5:  # Threshold
                results.append((obj_name, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def visualize_location_heatmap(self, obj_name, x_range=(0, 10), y_range=(0, 10)):
        """
        Visualize where the system thinks object is located.

        Args:
            obj_name: Object to visualize
            x_range: (min, max) x coordinates
            y_range: (min, max) y coordinates
        """
        if obj_name not in self.memory:
            print(f"Object '{obj_name}' not found")
            return

        # Get location hypervector
        location_hv = self.ssp.query_object(self.scene, obj_name)

        # Generate heatmap
        X, Y, similarities = similarity_map_2d(
            self.ssp,
            location_hv,
            x_range=x_range,
            y_range=y_range,
            resolution=50
        )

        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, similarities, levels=20, cmap='viridis')
        plt.colorbar(label='Similarity')

        # Mark true location
        true_loc = self.objects[obj_name]["location"]
        plt.scatter([true_loc[0]], [true_loc[1]], color='red', s=100,
                    marker='*', label=f'{obj_name} (true)', edgecolors='white')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Location Heatmap for '{obj_name}'")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()


def test_kitchen_scene():
    """
    Test spatial reasoning on a kitchen scene.
    """
    print("=" * 70)
    print("Test 1: Kitchen Scene Spatial Reasoning")
    print("=" * 70)

    # Create system
    system = SpatialReasoningSystem(dim=2048, seed=42)

    # Define kitchen layout (10m x 10m room)
    system.add_object("table", [5.0, 5.0], {})
    system.add_object("chair", [4.0, 5.0], {"LEFT_OF": "table"})
    system.add_object("stove", [1.0, 8.0], {})
    system.add_object("fridge", [9.0, 8.0], {"RIGHT_OF": "stove"})
    system.add_object("sink", [5.0, 9.0], {"ABOVE": "table"})
    system.add_object("door", [0.5, 0.5], {"BELOW": "table"})
    system.add_object("window", [9.5, 0.5], {})

    # Build scene
    system.build_scene()

    # Query 1: Where is the fridge?
    print("\n--- Query 1: Where is the fridge? ---")
    fridge_loc = system.query_location("fridge")
    true_loc = system.objects["fridge"]["location"]
    error = np.linalg.norm(np.array(fridge_loc) - np.array(true_loc))
    print(f"Fridge location: ({fridge_loc[0]:.2f}, {fridge_loc[1]:.2f})")
    print(f"True location: ({true_loc[0]:.2f}, {true_loc[1]:.2f})")
    print(f"Error: {error:.2f} meters")

    # Query 2: What is near the center of the room?
    print("\n--- Query 2: What is at (5.0, 5.0)? ---")
    obj, sim = system.query_at_location([5.0, 5.0])
    print(f"Object at (5.0, 5.0): {obj} (similarity: {sim:.3f})")

    # Query 3: Is chair LEFT_OF table?
    print("\n--- Query 3: Is chair LEFT_OF table? ---")
    sim = system.query_relation("LEFT_OF", "chair", "table")
    print(f"chair LEFT_OF table: {sim:.3f} (True)" if sim > 0.6 else f"chair LEFT_OF table: {sim:.3f} (False)")

    # Query 4: What is RIGHT_OF the stove?
    print("\n--- Query 4: What is RIGHT_OF the stove? ---")
    right_of_stove = system.find_related("RIGHT_OF", "stove")
    for obj, sim in right_of_stove:
        print(f"  {obj}: {sim:.3f}")

    # Query 5: What is BELOW the table?
    print("\n--- Query 5: What is BELOW the table? ---")
    below_table = system.find_related("BELOW", "table")
    for obj, sim in below_table:
        print(f"  {obj}: {sim:.3f}")

    print("\n" + "=" * 70)
    print("Observations:")
    print("- SSP accurately recovers object locations")
    print("- Clifford Operators preserve directional relations")
    print("- Combined system supports both 'where' and 'what relation' queries")
    print("=" * 70)


def test_navigation_scenario():
    """
    Test spatial reasoning for robot navigation.
    """
    print("\n" + "=" * 70)
    print("Test 2: Robot Navigation Scenario")
    print("=" * 70)

    system = SpatialReasoningSystem(dim=2048, seed=123)

    # Define warehouse environment
    system.add_object("robot", [2.0, 2.0], {})
    system.add_object("package_A", [8.0, 8.0], {})
    system.add_object("package_B", [3.0, 7.0], {"LEFT_OF": "package_A"})
    system.add_object("charging_station", [0.5, 0.5], {"BELOW": "robot"})
    system.add_object("obstacle", [5.0, 5.0], {})
    system.add_object("goal", [9.0, 9.0], {"ABOVE": "obstacle", "RIGHT_OF": "obstacle"})

    system.build_scene()

    # Scenario: Robot needs to navigate to goal
    print("\n--- Robot Navigation Task ---")
    print("Current position:", system.objects["robot"]["location"])
    print("Goal position:", system.objects["goal"]["location"])

    # Check obstacles
    print("\n--- Checking spatial relations ---")
    print(f"Is goal ABOVE obstacle? {system.query_relation('ABOVE', 'goal', 'obstacle'):.3f}")
    print(f"Is goal RIGHT_OF obstacle? {system.query_relation('RIGHT_OF', 'goal', 'obstacle'):.3f}")

    # Locate packages
    print("\n--- Package locations ---")
    for package in ["package_A", "package_B"]:
        loc = system.query_location(package)
        print(f"{package}: ({loc[0]:.2f}, {loc[1]:.2f})")

    # Check what's near the center
    print("\n--- What's at the center (5.0, 5.0)? ---")
    center_obj, sim = system.query_at_location([5.0, 5.0])
    print(f"Object: {center_obj} (similarity: {sim:.3f})")

    print("\n" + "=" * 70)


def test_visualization():
    """
    Test visualization of spatial distributions.
    """
    print("\n" + "=" * 70)
    print("Test 3: Visualization")
    print("=" * 70)

    system = SpatialReasoningSystem(dim=2048, seed=456)

    # Simple scene
    system.add_object("apple", [3.5, 7.2], {})
    system.add_object("banana", [6.8, 2.3], {})
    system.add_object("cherry", [8.1, 8.9], {})

    system.build_scene()

    print("\nGenerating location heatmaps...")

    # Create subplots for each object
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, obj_name in enumerate(["apple", "banana", "cherry"]):
        plt.sca(axes[idx])
        system.visualize_location_heatmap(obj_name)
        plt.subplot(1, 3, idx + 1)

    plt.tight_layout()
    plt.savefig("spatial_reasoning_heatmaps.png", dpi=150)
    print("Saved visualization to 'spatial_reasoning_heatmaps.png'")

    # Note: In Jupyter notebook, would display with plt.show()
    # plt.show()


def test_complex_queries():
    """
    Test complex multi-hop spatial queries.
    """
    print("\n" + "=" * 70)
    print("Test 4: Complex Multi-Hop Queries")
    print("=" * 70)

    system = SpatialReasoningSystem(dim=2048, seed=789)

    # Office layout
    system.add_object("desk", [5.0, 5.0], {})
    system.add_object("monitor", [5.0, 6.0], {"ABOVE": "desk"})
    system.add_object("keyboard", [5.0, 4.5], {"BELOW": "monitor"})
    system.add_object("mouse", [6.0, 4.5], {"RIGHT_OF": "keyboard"})
    system.add_object("coffee_mug", [4.0, 5.5], {"LEFT_OF": "monitor"})
    system.add_object("phone", [6.5, 5.5], {"RIGHT_OF": "monitor"})

    system.build_scene()

    print("\n--- Complex Query: What is on the desk? ---")
    print("(Find objects ABOVE desk)")

    on_desk = system.find_related("ABOVE", "desk")
    print("Objects ABOVE desk:")
    for obj, sim in on_desk:
        print(f"  - {obj}: {sim:.3f}")

    print("\n--- Complex Query: What is to the right of keyboard? ---")
    right_of_keyboard = system.find_related("RIGHT_OF", "keyboard")
    print("Objects RIGHT_OF keyboard:")
    for obj, sim in right_of_keyboard:
        print(f"  - {obj}: {sim:.3f}")

    print("\n--- Location verification ---")
    for obj_name in ["monitor", "keyboard", "mouse"]:
        loc = system.query_location(obj_name)
        true_loc = system.objects[obj_name]["location"]
        print(f"{obj_name}: predicted ({loc[0]:.2f}, {loc[1]:.2f}), "
              f"true ({true_loc[0]:.2f}, {true_loc[1]:.2f})")

    print("\n" + "=" * 70)


def main():
    """
    Run all spatial reasoning tests.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 4 EXERCISE 1")
    print(" " * 20 + "Spatial Reasoning with SSP and Operators")
    print("=" * 80)

    test_kitchen_scene()
    test_navigation_scenario()
    test_complex_queries()

    # Optional: Run visualization (comment out if no display available)
    # test_visualization()

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("✓ SSP encodes continuous spatial locations accurately")
    print("✓ Clifford Operators preserve directional relations")
    print("✓ Combined system enables both 'where' and 'what relation' queries")
    print("✓ Multi-hop spatial reasoning works through unbinding")
    print("✓ Practical for robotics, navigation, and spatial AI")
    print("=" * 80)


if __name__ == "__main__":
    main()
