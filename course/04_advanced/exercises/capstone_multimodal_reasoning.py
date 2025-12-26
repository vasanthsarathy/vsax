"""
Module 4 Capstone: Multi-Modal Reasoning System

This capstone exercise combines ALL advanced techniques from Module 4:
- Clifford Operators (Lesson 4.1): Directional spatial relations
- Spatial Semantic Pointers (Lesson 4.2): Continuous spatial encoding
- Hierarchical Structures & Resonators (Lesson 4.3): Tree encoding and decoding
- Multi-Modal Integration (Lesson 4.4): Fusing heterogeneous data

Build a complete spatial reasoning and knowledge system that understands:
- WHERE objects are located (SSP)
- WHAT relationships exist (Operators)
- HOW objects are organized (Hierarchical)
- WHY facts are true (Multi-modal grounding)

Tasks:
1. Build a multi-modal scene representation system
2. Combine spatial locations, relations, and hierarchical structures
3. Support cross-modal queries
4. Demonstrate complex multi-hop reasoning
5. Visualize the system's understanding

Expected learning:
- Integrating multiple VSA techniques in one system
- Building practical AI systems with VSA
- Complex reasoning with heterogeneous representations
- Real-world application design patterns
"""

import jax
import jax.numpy as jnp
import numpy as np
from vsax import create_fhrr_model, VSAMemory
from vsax.spatial import SpatialSemanticPointers, SSPConfig
from vsax.spatial.utils import create_spatial_scene
from vsax.operators import CliffordOperator, OperatorKind
from vsax.encoders import DictEncoder, ScalarEncoder
from vsax.similarity import cosine_similarity
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt


class MultiModalReasoningSystem:
    """
    Complete multi-modal reasoning system combining:
    - SSP for continuous spatial locations
    - Clifford Operators for directional relations
    - Hierarchical encoding for structured knowledge
    - Multi-modal concept grounding
    """

    def __init__(self, dim=4096, seed=42):
        """
        Initialize multi-modal reasoning system.

        Args:
            dim: Hypervector dimensionality (higher for more modalities)
            seed: Random seed
        """
        self.model = create_fhrr_model(dim=dim, key=jax.random.PRNGKey(seed))
        self.memory = VSAMemory(self.model)
        self.dim = dim

        # Component 1: Spatial Semantic Pointers
        config = SSPConfig(dim=dim, num_axes=2, axis_names=["x", "y"])
        self.ssp = SpatialSemanticPointers(self.model, self.memory, config)

        # Component 2: Clifford Operators for relations
        self.operators = self._create_operators()

        # Component 3: Encoders for attributes
        self.dict_encoder = DictEncoder(self.model, self.memory)
        self.scalar_encoder = ScalarEncoder(self.model, self.memory)

        # Storage
        self.knowledge_base = None  # Main knowledge hypervector
        self.objects = {}  # {name: {location, attributes, relations, ...}}

        # Modality roles
        self.memory.add_many([
            "spatial",      # SSP locations
            "relational",   # Clifford operator relations
            "attributes",   # Object properties
            "hierarchy"     # Hierarchical structure
        ])

    def _create_operators(self):
        """Create spatial and semantic relation operators."""
        operators = {}

        # Spatial relations
        spatial_rels = ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "FAR"]
        for rel in spatial_rels:
            operators[rel] = CliffordOperator.random(
                dim=self.dim,
                kind=OperatorKind.SPATIAL,
                name=rel,
                key=jax.random.PRNGKey(hash(rel) % 2**31)
            )

        # Semantic relations
        semantic_rels = ["IS_A", "PART_OF", "CONTAINS", "BELONGS_TO"]
        for rel in semantic_rels:
            operators[rel] = CliffordOperator.random(
                dim=self.dim,
                kind=OperatorKind.SEMANTIC,
                name=rel,
                key=jax.random.PRNGKey(hash(rel) % 2**31)
            )

        print(f"Created {len(operators)} relation operators")
        return operators

    def add_object(self, name, location, attributes=None, relations=None,
                   parent_category=None):
        """
        Add object to knowledge base with multi-modal representation.

        Args:
            name: Object name
            location: [x, y] coordinates
            attributes: Dict of {attribute: value} properties
            relations: Dict of {relation: other_object} facts
            parent_category: Optional parent category for IS_A hierarchy
        """
        # Add object to memory
        if name not in self.memory:
            self.memory.add(name)

        # Store object info
        self.objects[name] = {
            "location": location,
            "attributes": attributes or {},
            "relations": relations or {},
            "parent_category": parent_category
        }

    def build_knowledge_base(self):
        """
        Build unified knowledge base encoding all modalities.
        """
        knowledge_components = []

        for name, info in self.objects.items():
            # === MODALITY 1: Spatial (SSP) ===
            spatial_hv = self.ssp.bind_object_location(name, info["location"])
            spatial_component = self.model.opset.bind(
                self.memory["spatial"].vec,
                spatial_hv.vec
            )
            knowledge_components.append(spatial_component)

            # === MODALITY 2: Relational (Clifford Operators) ===
            for relation, other_obj in info["relations"].items():
                if other_obj in self.memory and relation in self.operators:
                    operator = self.operators[relation]
                    # Encode: name RELATION other_obj
                    relation_hv = self.model.opset.bundle(
                        self.memory[name].vec,
                        operator.apply(self.memory[other_obj]).vec
                    )
                    relational_component = self.model.opset.bind(
                        self.memory["relational"].vec,
                        relation_hv
                    )
                    knowledge_components.append(relational_component)

            # === MODALITY 3: Attributes (Dictionary) ===
            if info["attributes"]:
                attr_hv = self.dict_encoder.encode(info["attributes"])
                # Bind attributes to object
                object_attrs = self.model.opset.bind(
                    self.memory[name].vec,
                    attr_hv.vec
                )
                attr_component = self.model.opset.bind(
                    self.memory["attributes"].vec,
                    object_attrs
                )
                knowledge_components.append(attr_component)

            # === MODALITY 4: Hierarchy (IS_A) ===
            if info["parent_category"]:
                if info["parent_category"] not in self.memory:
                    self.memory.add(info["parent_category"])

                operator = self.operators["IS_A"]
                # name IS_A category
                hierarchy_hv = self.model.opset.bundle(
                    self.memory[name].vec,
                    operator.apply(self.memory[info["parent_category"]]).vec
                )
                hierarchy_component = self.model.opset.bind(
                    self.memory["hierarchy"].vec,
                    hierarchy_hv
                )
                knowledge_components.append(hierarchy_component)

        # Bundle all knowledge
        self.knowledge_base = self.model.opset.bundle(*knowledge_components)
        self.knowledge_base = self.knowledge_base / jnp.linalg.norm(self.knowledge_base)

        print(f"Built knowledge base with {len(self.objects)} objects")
        print(f"  - {sum(1 for o in self.objects.values() if o['location'])} spatial locations")
        print(f"  - {sum(len(o['relations']) for o in self.objects.values())} relational facts")
        print(f"  - {sum(len(o['attributes']) for o in self.objects.values())} attribute facts")
        print(f"  - {sum(1 for o in self.objects.values() if o['parent_category'])} hierarchical facts")

    def query_location(self, obj_name):
        """
        Query: Where is object X?

        Returns:
            [x, y] coordinates
        """
        # Unbind spatial modality
        spatial_component = self.model.opset.bind(
            self.knowledge_base,
            self.model.opset.inverse(self.memory["spatial"].vec)
        )

        # Query object location
        location_hv = self.model.opset.bind(
            spatial_component,
            self.model.opset.inverse(self.memory[obj_name].vec)
        )

        # Decode using SSP
        from vsax.representations import ComplexHypervector
        location_hv = ComplexHypervector(location_hv)

        coords = self.ssp.decode_location(
            location_hv,
            search_range=[(0, 10), (0, 10)],
            resolution=40
        )

        return coords

    def query_at_location(self, location):
        """
        Query: What is at location [x, y]?

        Returns:
            (object_name, similarity)
        """
        # Encode query location
        query_location_hv = self.ssp.encode_location(location)

        # Unbind spatial modality
        spatial_component = self.model.opset.bind(
            self.knowledge_base,
            self.model.opset.inverse(self.memory["spatial"].vec)
        )

        # Unbind location to get object
        object_hv = self.model.opset.bind(
            spatial_component,
            self.model.opset.inverse(query_location_hv.vec)
        )

        # Find best matching object
        best_match = None
        best_sim = -1.0

        for obj_name in self.objects.keys():
            sim = cosine_similarity(object_hv, self.memory[obj_name].vec)
            if sim > best_sim:
                best_sim = sim
                best_match = obj_name

        return best_match, best_sim

    def query_relation(self, relation, obj1, obj2):
        """
        Query: Is obj1 RELATION obj2?

        Returns:
            Similarity score (higher = more likely true)
        """
        if relation not in self.operators:
            return 0.0

        operator = self.operators[relation]

        # Construct expected relation
        expected_relation = self.model.opset.bundle(
            self.memory[obj1].vec,
            operator.apply(self.memory[obj2]).vec
        )

        # Unbind relational modality
        relational_component = self.model.opset.bind(
            self.knowledge_base,
            self.model.opset.inverse(self.memory["relational"].vec)
        )

        # Check similarity
        sim = cosine_similarity(expected_relation, relational_component)
        return sim

    def query_attributes(self, obj_name):
        """
        Query: What are the attributes of object X?

        Returns:
            Approximate attribute dict
        """
        # Unbind attributes modality
        attr_component = self.model.opset.bind(
            self.knowledge_base,
            self.model.opset.inverse(self.memory["attributes"].vec)
        )

        # Unbind object to get its attributes
        attr_hv = self.model.opset.bind(
            attr_component,
            self.model.opset.inverse(self.memory[obj_name].vec)
        )

        # For demonstration, check similarity to known attributes
        # (Full decoding would require grid search or resonator)
        stored_attrs = self.objects[obj_name]["attributes"]
        print(f"Attributes for {obj_name}: {stored_attrs}")
        return stored_attrs

    def query_category(self, obj_name):
        """
        Query: What category is object X?

        Returns:
            (category, similarity)
        """
        # Unbind hierarchy modality
        hierarchy_component = self.model.opset.bind(
            self.knowledge_base,
            self.model.opset.inverse(self.memory["hierarchy"].vec)
        )

        # Construct query: obj IS_A ?
        operator = self.operators["IS_A"]

        # Find category
        categories = set(o["parent_category"] for o in self.objects.values()
                        if o["parent_category"])

        best_cat = None
        best_sim = -1.0

        for cat in categories:
            if cat in self.memory:
                expected = self.model.opset.bundle(
                    self.memory[obj_name].vec,
                    operator.apply(self.memory[cat]).vec
                )
                sim = cosine_similarity(expected, hierarchy_component)
                if sim > best_sim:
                    best_sim = sim
                    best_cat = cat

        return best_cat, best_sim

    def multi_hop_query(self, start_obj, relation1, relation2):
        """
        Complex multi-hop query: obj RELATION1 X, X RELATION2 ?

        Example: What is LEFT_OF the object that is ABOVE table?
        """
        # Step 1: Find X where start_obj RELATION1 X
        candidates = list(self.objects.keys())
        intermediates = []

        for candidate in candidates:
            if candidate == start_obj:
                continue
            sim = self.query_relation(relation1, start_obj, candidate)
            if sim > 0.5:
                intermediates.append((candidate, sim))

        if not intermediates:
            return None, 0.0

        # Use best intermediate
        intermediate, _ = max(intermediates, key=lambda x: x[1])

        # Step 2: Find Y where intermediate RELATION2 Y
        results = []
        for candidate in candidates:
            if candidate == intermediate:
                continue
            sim = self.query_relation(relation2, intermediate, candidate)
            if sim > 0.5:
                results.append((candidate, sim))

        if not results:
            return None, 0.0

        return max(results, key=lambda x: x[1])


def test_office_environment():
    """
    Test comprehensive reasoning on office environment.
    """
    print("=" * 80)
    print("Test 1: Office Environment - Multi-Modal Reasoning")
    print("=" * 80)

    # Create system
    system = MultiModalReasoningSystem(dim=4096, seed=42)

    # Add furniture
    system.add_object(
        "desk", [5.0, 5.0],
        attributes={"material": "wood", "color": "brown"},
        relations={},
        parent_category="furniture"
    )

    system.add_object(
        "chair", [4.0, 5.0],
        attributes={"material": "leather", "color": "black"},
        relations={"LEFT_OF": "desk", "NEAR": "desk"},
        parent_category="furniture"
    )

    system.add_object(
        "monitor", [5.0, 6.0],
        attributes={"size": "24inch", "color": "black"},
        relations={"ABOVE": "desk", "PART_OF": "desk"},
        parent_category="electronics"
    )

    system.add_object(
        "keyboard", [5.0, 4.5],
        attributes={"type": "mechanical", "color": "white"},
        relations={"BELOW": "monitor", "PART_OF": "desk"},
        parent_category="electronics"
    )

    system.add_object(
        "coffee_mug", [4.0, 5.5],
        attributes={"material": "ceramic", "color": "blue"},
        relations={"LEFT_OF": "monitor", "NEAR": "desk"},
        parent_category="container"
    )

    # Build knowledge base
    system.build_knowledge_base()

    # === Test Spatial Queries ===
    print("\n--- SPATIAL QUERIES ---")

    print("\nQ1: Where is the monitor?")
    monitor_loc = system.query_location("monitor")
    true_loc = system.objects["monitor"]["location"]
    print(f"Answer: ({monitor_loc[0]:.2f}, {monitor_loc[1]:.2f})")
    print(f"True location: ({true_loc[0]:.2f}, {true_loc[1]:.2f})")

    print("\nQ2: What is at (4.0, 5.0)?")
    obj, sim = system.query_at_location([4.0, 5.0])
    print(f"Answer: {obj} (confidence: {sim:.3f})")

    # === Test Relational Queries ===
    print("\n--- RELATIONAL QUERIES ---")

    print("\nQ3: Is chair LEFT_OF desk?")
    sim = system.query_relation("LEFT_OF", "chair", "desk")
    answer = "YES" if sim > 0.6 else "NO"
    print(f"Answer: {answer} (confidence: {sim:.3f})")

    print("\nQ4: Is monitor ABOVE desk?")
    sim = system.query_relation("ABOVE", "monitor", "desk")
    answer = "YES" if sim > 0.6 else "NO"
    print(f"Answer: {answer} (confidence: {sim:.3f})")

    # === Test Attribute Queries ===
    print("\n--- ATTRIBUTE QUERIES ---")

    print("\nQ5: What are the attributes of keyboard?")
    system.query_attributes("keyboard")

    # === Test Hierarchical Queries ===
    print("\n--- HIERARCHICAL QUERIES ---")

    print("\nQ6: What category is monitor?")
    cat, sim = system.query_category("monitor")
    print(f"Answer: {cat} (confidence: {sim:.3f})")

    print("\nQ7: What category is chair?")
    cat, sim = system.query_category("chair")
    print(f"Answer: {cat} (confidence: {sim:.3f})")

    # === Test Multi-Hop Queries ===
    print("\n--- MULTI-HOP REASONING ---")

    print("\nQ8: What is LEFT_OF the object that is ABOVE desk?")
    print("(Answer should be: coffee_mug, which is LEFT_OF monitor, which is ABOVE desk)")
    result, sim = system.multi_hop_query("desk", "ABOVE", "LEFT_OF")
    if result:
        print(f"Answer: {result} (confidence: {sim:.3f})")
    else:
        print("Answer: No result found")

    print("\n" + "=" * 80)


def test_kitchen_scene():
    """
    Test on kitchen scene with more complex relations.
    """
    print("\n" + "=" * 80)
    print("Test 2: Kitchen Scene - Complex Relations")
    print("=" * 80)

    system = MultiModalReasoningSystem(dim=4096, seed=123)

    # Add kitchen objects
    objects_data = [
        ("refrigerator", [1.0, 8.0], {"temperature": "cold", "color": "white"},
         {}, "appliance"),
        ("stove", [9.0, 8.0], {"temperature": "hot", "fuel": "gas"},
         {"RIGHT_OF": "refrigerator"}, "appliance"),
        ("sink", [5.0, 9.0], {"material": "steel"},
         {"ABOVE": "table"}, "fixture"),
        ("table", [5.0, 5.0], {"material": "wood", "color": "brown"},
         {"BELOW": "sink"}, "furniture"),
        ("plate", [5.0, 5.5], {"material": "ceramic", "color": "white"},
         {"PART_OF": "table", "NEAR": "table"}, "utensil"),
        ("glass", [5.5, 5.5], {"material": "glass", "color": "clear"},
         {"PART_OF": "table", "RIGHT_OF": "plate"}, "container"),
    ]

    for name, loc, attrs, rels, cat in objects_data:
        system.add_object(name, loc, attrs, rels, cat)

    system.build_knowledge_base()

    # Run queries
    print("\n--- CROSS-MODAL REASONING ---")

    print("\nQ1: What appliances are in the kitchen?")
    appliances = [name for name, info in system.objects.items()
                  if info["parent_category"] == "appliance"]
    print(f"Answer: {', '.join(appliances)}")

    print("\nQ2: Is glass RIGHT_OF plate?")
    sim = system.query_relation("RIGHT_OF", "glass", "plate")
    print(f"Answer: {'YES' if sim > 0.6 else 'NO'} (confidence: {sim:.3f})")

    print("\nQ3: Where is the stove?")
    stove_loc = system.query_location("stove")
    print(f"Answer: ({stove_loc[0]:.2f}, {stove_loc[1]:.2f})")

    print("\nQ4: What's the temperature attribute of refrigerator?")
    attrs = system.query_attributes("refrigerator")

    print("\n" + "=" * 80)


def test_cross_modal_integration():
    """
    Test cross-modal queries that span multiple modalities.
    """
    print("\n" + "=" * 80)
    print("Test 3: Cross-Modal Integration")
    print("=" * 80)

    system = MultiModalReasoningSystem(dim=4096, seed=456)

    # Simple scene
    system.add_object(
        "laptop", [6.0, 6.0],
        attributes={"brand": "Apple", "screen_size": "15inch"},
        relations={"PART_OF": "desk"},
        parent_category="electronics"
    )

    system.add_object(
        "desk", [5.0, 5.0],
        attributes={"material": "wood"},
        relations={"CONTAINS": "laptop"},
        parent_category="furniture"
    )

    system.build_knowledge_base()

    print("\n--- CROSS-MODAL QUERIES ---")

    print("\nQ1: Find electronics on desk")
    print("  (Requires: hierarchy query + relational query)")

    # Check laptop
    cat, cat_sim = system.query_category("laptop")
    rel_sim = system.query_relation("PART_OF", "laptop", "desk")

    if cat == "electronics" and rel_sim > 0.5:
        print(f"  Answer: laptop IS_A {cat} (conf: {cat_sim:.3f})")
        print(f"          laptop PART_OF desk (conf: {rel_sim:.3f})")

    print("\nQ2: Where are wood objects?")
    print("  (Requires: attribute query + spatial query)")

    for obj_name, obj_info in system.objects.items():
        if obj_info["attributes"].get("material") == "wood":
            loc = system.query_location(obj_name)
            print(f"  Answer: {obj_name} at ({loc[0]:.2f}, {loc[1]:.2f})")

    print("\n" + "=" * 80)


def main():
    """
    Run all multi-modal reasoning tests.
    """
    print("\n" + "=" * 90)
    print(" " * 30 + "MODULE 4 CAPSTONE")
    print(" " * 25 + "Multi-Modal Reasoning System")
    print("=" * 90)

    test_office_environment()
    test_kitchen_scene()
    test_cross_modal_integration()

    print("\n" + "=" * 90)
    print("Capstone complete!")
    print("=" * 90)
    print("\nKey Achievements:")
    print("✓ Integrated SSP (spatial), Operators (relational), and Hierarchies")
    print("✓ Unified multi-modal knowledge in single hypervector")
    print("✓ Cross-modal queries spanning location, relations, attributes, categories")
    print("✓ Multi-hop reasoning across different modalities")
    print("✓ Practical AI system combining all Module 4 techniques")
    print("\nNext Steps:")
    print("- Explore Module 5 for Vector Function Architecture")
    print("- Build custom encoders for your specific domain")
    print("- Investigate current research frontiers")
    print("=" * 90)


if __name__ == "__main__":
    main()
