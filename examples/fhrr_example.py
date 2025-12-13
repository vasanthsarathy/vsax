"""Complete FHRR model example demonstrating all encoders.

This example shows how to use VSAX with the FHRR model (complex hypervectors)
to encode various data types: scalars, sequences, sets, dictionaries, and graphs.
"""

from vsax import (
    create_fhrr_model,
    VSAMemory,
    ScalarEncoder,
    SequenceEncoder,
    SetEncoder,
    DictEncoder,
    GraphEncoder,
)


def main():
    """Demonstrate FHRR model with all encoder types."""
    print("=" * 60)
    print("VSAX FHRR Model Example")
    print("=" * 60)

    # Create FHRR model (complex hypervectors, FFT-based operations)
    print("\n1. Creating FHRR model...")
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    print(f"   Model created with dimension: {model.dim}")
    print(f"   Representation: {model.rep_cls.__name__}")
    print(f"   Operations: {model.opset.__class__.__name__}")

    # Add basis symbols to memory
    print("\n2. Adding symbols to memory...")
    symbols = [
        # For scalar encoding
        "temperature", "humidity",
        # For sequence encoding
        "red", "green", "blue",
        # For set encoding
        "dog", "cat", "bird",
        # For dict encoding
        "subject", "action", "object", "run", "jump", "ball",
        # For graph encoding
        "Alice", "Bob", "Charlie", "knows", "likes",
    ]
    memory.add_many(symbols)
    print(f"   Added {len(memory)} symbols to memory")

    # Scalar Encoding Example
    print("\n3. Scalar Encoding (temperature and humidity)...")
    scalar_encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)
    temp_hv = scalar_encoder.encode("temperature", 23.5)
    humidity_hv = scalar_encoder.encode("humidity", 65.0)
    print(f"   Temperature (23.5°C) encoded: {temp_hv.vec.shape}")
    print(f"   Humidity (65%) encoded: {humidity_hv.vec.shape}")

    # Sequence Encoding Example
    print("\n4. Sequence Encoding (ordered colors)...")
    seq_encoder = SequenceEncoder(model, memory)
    colors_seq = seq_encoder.encode(["red", "green", "blue"])
    colors_reversed = seq_encoder.encode(["blue", "green", "red"])
    print(f"   Sequence [red, green, blue] encoded: {colors_seq.vec.shape}")
    print(f"   Different order produces different hypervector")

    # Set Encoding Example
    print("\n5. Set Encoding (unordered animals)...")
    set_encoder = SetEncoder(model, memory)
    animals_set = set_encoder.encode({"dog", "cat", "bird"})
    print(f"   Set {{dog, cat, bird}} encoded: {animals_set.vec.shape}")
    print(f"   Order doesn't matter in sets")

    # Dictionary Encoding Example
    print("\n6. Dictionary Encoding (sentence structure)...")
    dict_encoder = DictEncoder(model, memory)
    sentence1 = dict_encoder.encode({
        "subject": "dog",
        "action": "run",
    })
    sentence2 = dict_encoder.encode({
        "subject": "cat",
        "action": "jump",
        "object": "ball",
    })
    print(f"   'dog run' encoded: {sentence1.vec.shape}")
    print(f"   'cat jump ball' encoded: {sentence2.vec.shape}")

    # Graph Encoding Example
    print("\n7. Graph Encoding (social network)...")
    graph_encoder = GraphEncoder(model, memory)
    social_graph = graph_encoder.encode([
        ("Alice", "knows", "Bob"),
        ("Alice", "likes", "Charlie"),
        ("Bob", "knows", "Charlie"),
    ])
    print(f"   Social network encoded: {social_graph.vec.shape}")
    print(f"   3 edges (relationships) bundled together")

    # Demonstrate binding and bundling
    print("\n8. Combining hypervectors...")
    # Bind two concepts together
    dog_runs = model.opset.bind(memory["dog"].vec, memory["run"].vec)
    print(f"   'dog' BIND 'run': creates new concept")

    # Bundle multiple concepts
    all_animals = model.opset.bundle(
        memory["dog"].vec,
        memory["cat"].vec,
        memory["bird"].vec,
    )
    print(f"   bundle(dog, cat, bird): creates superposition")

    print("\n" + "=" * 60)
    print("FHRR Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
