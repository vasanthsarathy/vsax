"""Basic binding and bundling example.

This simple example demonstrates the fundamental VSA operations:
binding (associating concepts) and bundling (creating superpositions).
"""

from vsax import create_fhrr_model, VSAMemory


def main():
    """Demonstrate basic binding and bundling operations."""
    print("=" * 60)
    print("Basic VSA Operations: Binding and Bundling")
    print("=" * 60)

    # Create model and memory
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)

    # Add some concepts
    print("\n1. Adding concepts to memory...")
    memory.add_many(["red", "square", "large", "blue", "circle", "small"])
    print(f"   Memory contains: {list(memory.keys())}")

    # Binding: Associate two concepts
    print("\n2. Binding: Create composite concepts...")
    red_square = model.opset.bind(
        memory["red"].vec,
        memory["square"].vec
    )
    print(f"   red BIND square = red_square")
    print(f"   Creates a hypervector representing 'red square'")

    blue_circle = model.opset.bind(
        memory["blue"].vec,
        memory["circle"].vec
    )
    print(f"   blue BIND circle = blue_circle")

    # Bundling: Create superposition
    print("\n3. Bundling: Create superposition...")
    shapes = model.opset.bundle(
        red_square,
        blue_circle
    )
    print(f"   red_square BUNDLE blue_circle = shapes")
    print(f"   Creates a hypervector representing BOTH shapes")

    # More complex example
    print("\n4. Combining operations...")
    large_red_square = model.opset.bind(
        memory["large"].vec,
        model.opset.bind(memory["red"].vec, memory["square"].vec)
    )
    print(f"   large BIND (red BIND square) = large_red_square")

    small_blue_circle = model.opset.bind(
        memory["small"].vec,
        model.opset.bind(memory["blue"].vec, memory["circle"].vec)
    )
    print(f"   small BIND (blue BIND circle) = small_blue_circle")

    all_objects = model.opset.bundle(large_red_square, small_blue_circle)
    print(f"   Bundle both objects into a single hypervector")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Binding: Associates concepts (like multiplication)")
    print("  - Bundling: Creates superpositions (like addition)")
    print("  - These operations are the foundation of all VSA encoders")
    print("=" * 60)


if __name__ == "__main__":
    main()
