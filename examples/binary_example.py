"""Complete Binary model example demonstrating all encoders.

This example shows how to use VSAX with the Binary model (discrete hypervectors)
to encode various data types efficiently.
"""

from vsax import (
    create_binary_model,
    VSAMemory,
    SequenceEncoder,
    SetEncoder,
    DictEncoder,
    GraphEncoder,
)


def main():
    """Demonstrate Binary model with encoders."""
    print("=" * 60)
    print("VSAX Binary Model Example")
    print("=" * 60)

    # Create Binary model (bipolar: -1, +1)
    print("\n1. Creating Binary model...")
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    print(f"   Model created with dimension: {model.dim}")
    print(f"   Representation: {model.rep_cls.__name__}")
    print(f"   Operations: {model.opset.__class__.__name__}")
    print(f"   Values: Bipolar {{-1, +1}}")

    # Example: Encoding a simple ontology
    print("\n2. Building an ontology...")

    # Add concepts
    concepts = [
        "animal", "plant", "mammal", "bird", "tree", "flower",
        "dog", "cat", "eagle", "oak", "rose",
        "has_property", "is_a",
    ]
    memory.add_many(concepts)
    print(f"   Added {len(memory)} concepts")

    # Encode taxonomic relationships
    print("\n3. Encoding taxonomy with GraphEncoder...")
    graph_encoder = GraphEncoder(model, memory)

    taxonomy = graph_encoder.encode([
        ("mammal", "is_a", "animal"),
        ("bird", "is_a", "animal"),
        ("tree", "is_a", "plant"),
        ("dog", "is_a", "mammal"),
        ("cat", "is_a", "mammal"),
        ("eagle", "is_a", "bird"),
        ("oak", "is_a", "tree"),
        ("rose", "is_a", "flower"),
    ])
    print(f"   Taxonomy graph encoded with 8 relationships")

    # Encode categories
    print("\n4. Encoding categories with SetEncoder...")
    set_encoder = SetEncoder(model, memory)

    mammals = set_encoder.encode({"dog", "cat"})
    birds = set_encoder.encode({"eagle"})
    trees = set_encoder.encode({"oak"})

    print(f"   Mammals set: {{dog, cat}}")
    print(f"   Birds set: {{eagle}}")
    print(f"   Trees set: {{oak}}")

    # Encode structured data
    print("\n5. Encoding structured data with DictEncoder...")
    dict_encoder = DictEncoder(model, memory)

    memory.add_many(["name", "category", "living"])

    dog_data = dict_encoder.encode({
        "name": "dog",
        "category": "mammal",
    })
    print(f"   Dog data: {{name: dog, category: mammal}}")

    # Encode sequences
    print("\n6. Encoding sequences with SequenceEncoder...")
    memory.add_many(["first", "second", "third"])

    seq_encoder = SequenceEncoder(model, memory)
    classification = seq_encoder.encode(["animal", "mammal", "dog"])
    print(f"   Classification hierarchy: [animal, mammal, dog]")

    # Demonstrate Binary-specific operations
    print("\n7. Binary Operations (XOR and Majority)...")
    dog_hv = memory["dog"]
    cat_hv = memory["cat"]

    # Binding in Binary: XOR (self-inverse)
    bound = model.opset.bind(dog_hv.vec, cat_hv.vec)
    print(f"   dog BIND cat (XOR binding)")

    # Unbinding: bind again with same vector
    unbound = model.opset.bind(bound, cat_hv.vec)
    print(f"   (dog BIND cat) BIND cat = dog (unbinding)")

    # Bundling in Binary: majority vote
    bundled = model.opset.bundle(dog_hv.vec, cat_hv.vec, memory["eagle"].vec)
    print(f"   bundle(dog, cat, eagle) via majority vote")

    print("\n" + "=" * 60)
    print("Binary Example Complete!")
    print("Binary models are memory-efficient and support exact unbinding")
    print("=" * 60)


if __name__ == "__main__":
    main()
