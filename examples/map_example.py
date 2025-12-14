"""Complete MAP model example demonstrating all encoders.

This example shows how to use VSAX with the MAP model (real-valued hypervectors)
to encode various data types and perform VSA operations.
"""

from vsax import (
    DictEncoder,
    ScalarEncoder,
    SequenceEncoder,
    VSAMemory,
    create_map_model,
)


def main():
    """Demonstrate MAP model with encoders."""
    print("=" * 60)
    print("VSAX MAP Model Example")
    print("=" * 60)

    # Create MAP model (real-valued hypervectors)
    print("\n1. Creating MAP model...")
    model = create_map_model(dim=1024)
    memory = VSAMemory(model)
    print(f"   Model created with dimension: {model.dim}")
    print(f"   Representation: {model.rep_cls.__name__}")
    print(f"   Operations: {model.opset.__class__.__name__}")

    # Example: Encoding a knowledge base
    print("\n2. Building a simple knowledge base...")

    # Add entities and relations
    entities = ["Python", "Java", "JavaScript", "programming", "language", "web"]
    relations = ["is_a", "used_for"]
    memory.add_many(entities + relations)
    print(f"   Added {len(memory)} symbols")

    # Encode facts using DictEncoder
    dict_encoder = DictEncoder(model, memory)

    fact1 = dict_encoder.encode({
        "subject": "Python",
        "relation": "is_a",
        "object": "programming",
    })
    print("\n   Fact 1: Python is_a programming language")

    fact2 = dict_encoder.encode({
        "subject": "JavaScript",
        "relation": "used_for",
        "object": "web",
    })
    print("   Fact 2: JavaScript used_for web development")

    # Encode numeric properties
    print("\n3. Encoding numeric properties...")
    memory.add_many(["version", "popularity"])

    scalar_encoder = ScalarEncoder(model, memory, min_val=0, max_val=10)
    python_version = scalar_encoder.encode("version", 3.11)
    python_popularity = scalar_encoder.encode("popularity", 9.5)
    print("   Python version 3.11 encoded")
    print("   Python popularity 9.5/10 encoded")

    # Encode ordered lists
    print("\n4. Encoding technology stacks...")
    memory.add_many(["frontend", "backend", "database", "React", "Django", "PostgreSQL"])

    seq_encoder = SequenceEncoder(model, memory)
    tech_stack = seq_encoder.encode(["React", "Django", "PostgreSQL"])
    print("   Tech stack [React, Django, PostgreSQL] encoded")
    print("   Order preserved in encoding")

    # Demonstrate MAP-specific operations
    print("\n5. MAP Operations (element-wise)...")
    python_hv = memory["Python"]
    java_hv = memory["Java"]

    # Binding in MAP: element-wise multiplication
    bound = model.opset.bind(python_hv.vec, java_hv.vec)
    print("   Python BIND Java (element-wise multiply)")

    # Bundling in MAP: element-wise mean
    bundled = model.opset.bundle(python_hv.vec, java_hv.vec)
    print("   Python BUNDLE Java (element-wise mean)")

    print("\n" + "=" * 60)
    print("MAP Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
