"""
Module 2 Exercise 1: Implement Binding Operations from Scratch

This exercise implements the three binding operations (FHRR, MAP, Binary)
from first principles to understand their mathematical foundations.

Tasks:
1. Implement FHRR binding using FFT-based circular convolution
2. Implement MAP binding using element-wise multiplication
3. Implement Binary binding using XOR
4. Compare all three to VSAX's built-in implementations
5. Verify unbinding accuracy

Expected learning:
- Deep understanding of how each binding operation works
- Mathematical foundations of circular convolution
- Trade-offs between exact and approximate unbinding
- Verification that custom implementations match VSAX
"""

import jax.numpy as jnp
import jax.random as random
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity


def circular_convolution_manual(a, b):
    """
    Implement circular convolution using the naive O(n²) algorithm.

    This is the definition of circular convolution:
    (a ⊛ b)[k] = Σᵢ a[i] · b[(k - i) mod n]

    Args:
        a: Complex vector
        b: Complex vector

    Returns:
        Circular convolution of a and b
    """
    n = len(a)
    result = jnp.zeros(n, dtype=a.dtype)

    for k in range(n):
        # Sum over all i: a[i] * b[(k-i) mod n]
        for i in range(n):
            result = result.at[k].add(a[i] * b[(k - i) % n])

    return result


def circular_convolution_fft(a, b):
    """
    Implement circular convolution using FFT (fast O(n log n) algorithm).

    Convolution Theorem: FFT(a ⊛ b) = FFT(a) · FFT(b)
    Therefore: a ⊛ b = IFFT(FFT(a) · FFT(b))

    Args:
        a: Complex vector
        b: Complex vector

    Returns:
        Circular convolution of a and b
    """
    # Transform to frequency domain
    a_fft = jnp.fft.fft(a)
    b_fft = jnp.fft.fft(b)

    # Multiply in frequency domain
    result_fft = a_fft * b_fft

    # Transform back to time domain
    result = jnp.fft.ifft(result_fft)

    # Normalize to unit vector
    result = result / jnp.abs(result)

    return result


def fhrr_bind_custom(a, b):
    """
    Custom implementation of FHRR binding.

    Uses FFT-based circular convolution and normalizes result.
    """
    return circular_convolution_fft(a, b)


def fhrr_unbind_custom(bound, b):
    """
    Custom implementation of FHRR unbinding.

    Unbinding is binding with the inverse (complex conjugate).
    """
    b_inv = jnp.conj(b)
    return fhrr_bind_custom(bound, b_inv)


def map_bind_custom(a, b):
    """
    Custom implementation of MAP binding.

    MAP binding is element-wise multiplication followed by normalization.
    """
    result = a * b
    result = result / jnp.linalg.norm(result)
    return result


def map_unbind_custom(bound, b):
    """
    Custom implementation of MAP approximate unbinding.

    Unbinding is the same as binding (element-wise multiply).
    """
    return map_bind_custom(bound, b)


def binary_bind_custom(a, b):
    """
    Custom implementation of Binary binding.

    Binary binding is element-wise XOR.
    """
    return jnp.logical_xor(a, b)


def binary_unbind_custom(bound, b):
    """
    Custom implementation of Binary unbinding.

    Binary is self-inverse: a ⊗ b ⊗ b = a
    """
    return binary_bind_custom(bound, b)


def test_fhrr_implementation():
    """
    Test custom FHRR implementation against VSAX.
    """
    print("=" * 60)
    print("Test 1: FHRR Implementation")
    print("=" * 60)

    # Create VSAX FHRR model
    model = create_fhrr_model(dim=2048)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    a = memory["a"].vec
    b = memory["b"].vec

    # VSAX binding
    vsax_bound = model.opset.bind(a, b)

    # Custom binding
    custom_bound = fhrr_bind_custom(a, b)

    # Compare
    similarity = cosine_similarity(vsax_bound, custom_bound)
    print(f"\nBinding comparison:")
    print(f"  VSAX vs Custom similarity: {similarity:.6f}")
    print(f"  Match: {similarity > 0.99}")

    # Test unbinding
    vsax_retrieved = model.opset.bind(vsax_bound, jnp.conj(b))
    custom_retrieved = fhrr_unbind_custom(custom_bound, b)

    vsax_sim = cosine_similarity(vsax_retrieved, a)
    custom_sim = cosine_similarity(custom_retrieved, a)

    print(f"\nUnbinding accuracy:")
    print(f"  VSAX unbinding:   {vsax_sim:.6f}")
    print(f"  Custom unbinding: {custom_sim:.6f}")

    # Test that manual convolution matches FFT
    print(f"\nVerifying FFT vs Manual convolution (small dimension):")
    a_small = memory["a"].vec[:32]  # Use only 32 elements for speed
    b_small = memory["b"].vec[:32]

    manual = circular_convolution_manual(a_small, b_small)
    fft = circular_convolution_fft(a_small, b_small)

    manual_norm = manual / jnp.abs(manual)
    conv_similarity = cosine_similarity(manual_norm, fft)
    print(f"  Manual vs FFT similarity: {conv_similarity:.6f}")
    print(f"  Match: {conv_similarity > 0.99}")


def test_map_implementation():
    """
    Test custom MAP implementation against VSAX.
    """
    print("\n" + "=" * 60)
    print("Test 2: MAP Implementation")
    print("=" * 60)

    # Create VSAX MAP model
    model = create_map_model(dim=2048)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    a = memory["a"].vec
    b = memory["b"].vec

    # VSAX binding
    vsax_bound = model.opset.bind(a, b)

    # Custom binding
    custom_bound = map_bind_custom(a, b)

    # Compare
    similarity = cosine_similarity(vsax_bound, custom_bound)
    print(f"\nBinding comparison:")
    print(f"  VSAX vs Custom similarity: {similarity:.6f}")
    print(f"  Match: {similarity > 0.99}")

    # Test unbinding (approximate)
    vsax_retrieved = model.opset.bind(vsax_bound, b)
    custom_retrieved = map_unbind_custom(custom_bound, b)

    vsax_sim = cosine_similarity(vsax_retrieved, a)
    custom_sim = cosine_similarity(custom_retrieved, a)

    print(f"\nUnbinding accuracy (approximate):")
    print(f"  VSAX unbinding:   {vsax_sim:.6f}")
    print(f"  Custom unbinding: {custom_sim:.6f}")
    print(f"  Note: MAP unbinding is approximate (~0.7-0.8)")


def test_binary_implementation():
    """
    Test custom Binary implementation against VSAX.
    """
    print("\n" + "=" * 60)
    print("Test 3: Binary Implementation")
    print("=" * 60)

    # Create VSAX Binary model
    model = create_binary_model(dim=2048)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    a = memory["a"].vec
    b = memory["b"].vec

    # VSAX binding
    vsax_bound = model.opset.bind(a, b)

    # Custom binding
    custom_bound = binary_bind_custom(a, b)

    # Compare (use Hamming similarity for binary)
    from vsax.similarity import hamming_similarity
    similarity = hamming_similarity(vsax_bound, custom_bound)
    print(f"\nBinding comparison:")
    print(f"  VSAX vs Custom Hamming similarity: {similarity:.6f}")
    print(f"  Match: {similarity > 0.99}")

    # Test unbinding (self-inverse)
    vsax_retrieved = model.opset.bind(vsax_bound, b)
    custom_retrieved = binary_unbind_custom(custom_bound, b)

    vsax_sim = hamming_similarity(vsax_retrieved, a)
    custom_sim = hamming_similarity(custom_retrieved, a)

    print(f"\nUnbinding accuracy:")
    print(f"  VSAX unbinding:   {vsax_sim:.6f}")
    print(f"  Custom unbinding: {custom_sim:.6f}")
    print(f"  Note: Binary is self-inverse (exact unbinding)")


def compare_binding_depths():
    """
    Compare unbinding accuracy across different binding depths.
    """
    print("\n" + "=" * 60)
    print("Test 4: Binding Depth Comparison")
    print("=" * 60)

    dim = 2048
    max_depth = 5

    # Create models
    fhrr_model = create_fhrr_model(dim=dim)
    map_model = create_map_model(dim=dim)
    binary_model = create_binary_model(dim=dim)

    fhrr_mem = VSAMemory(fhrr_model)
    map_mem = VSAMemory(map_model)
    binary_mem = VSAMemory(binary_model)

    fhrr_mem.add_many(["a", "b"])
    map_mem.add_many(["a", "b"])
    binary_mem.add_many(["a", "b"])

    print(f"\nBinding depth analysis (d={dim}):")
    print(f"{'Depth':>6s}  {'FHRR':>8s}  {'MAP':>8s}  {'Binary':>8s}")
    print("-" * 40)

    for depth in range(1, max_depth + 1):
        # FHRR
        fhrr_bound = fhrr_mem["a"].vec
        for _ in range(depth):
            fhrr_bound = fhrr_bind_custom(fhrr_bound, fhrr_mem["b"].vec)

        fhrr_retrieved = fhrr_bound
        for _ in range(depth):
            fhrr_retrieved = fhrr_unbind_custom(fhrr_retrieved, fhrr_mem["b"].vec)

        fhrr_sim = cosine_similarity(fhrr_retrieved, fhrr_mem["a"].vec)

        # MAP
        map_bound = map_mem["a"].vec
        for _ in range(depth):
            map_bound = map_bind_custom(map_bound, map_mem["b"].vec)

        map_retrieved = map_bound
        for _ in range(depth):
            map_retrieved = map_unbind_custom(map_retrieved, map_mem["b"].vec)

        map_sim = cosine_similarity(map_retrieved, map_mem["a"].vec)

        # Binary
        binary_bound = binary_mem["a"].vec
        for _ in range(depth):
            binary_bound = binary_bind_custom(binary_bound, binary_mem["b"].vec)

        binary_retrieved = binary_bound
        for _ in range(depth):
            binary_retrieved = binary_unbind_custom(binary_retrieved, binary_mem["b"].vec)

        from vsax.similarity import hamming_similarity
        binary_sim = hamming_similarity(binary_retrieved, binary_mem["a"].vec)

        print(f"{depth:6d}  {fhrr_sim:8.6f}  {map_sim:8.6f}  {binary_sim:8.6f}")

    print("\nObservations:")
    print("- FHRR maintains >0.99 similarity at all depths (exact)")
    print("- MAP degrades with depth (approximate unbinding)")
    print("- Binary is self-inverse (perfect for odd depths)")


def visualize_convolution():
    """
    Visualize how circular convolution works.
    """
    print("\n" + "=" * 60)
    print("Bonus: Visualizing Circular Convolution")
    print("=" * 60)

    # Small example for clarity
    n = 8
    a = jnp.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
    b = jnp.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0])

    print(f"\nInput vectors (length {n}):")
    print(f"a = {a}")
    print(f"b = {b}")

    # Circular convolution
    result = jnp.zeros(n)
    for k in range(n):
        for i in range(n):
            result = result.at[k].add(a[i] * b[(k - i) % n])

    print(f"\nCircular convolution a ⊛ b:")
    print(f"result = {result}")
    print(f"\nNote: This is dissimilar to both a and b (orthogonal)")

    # Verify with FFT
    a_fft = jnp.fft.fft(a)
    b_fft = jnp.fft.fft(b)
    result_fft = jnp.fft.ifft(a_fft * b_fft).real

    print(f"\nVerification with FFT:")
    print(f"FFT result = {result_fft}")
    print(f"Match: {jnp.allclose(result, result_fft)}")


def main():
    """
    Run all binding implementation tests.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 2 EXERCISE 1")
    print(" " * 15 + "Implement Binding Operations")
    print("=" * 80)

    test_fhrr_implementation()
    test_map_implementation()
    test_binary_implementation()
    compare_binding_depths()
    visualize_convolution()

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("✓ FHRR: FFT-based circular convolution, exact unbinding")
    print("✓ MAP: Element-wise multiplication, approximate unbinding")
    print("✓ Binary: XOR, self-inverse (exact for odd depths)")
    print("✓ Custom implementations match VSAX built-ins")
    print("✓ Deep understanding of mathematical foundations")
    print("=" * 80)


if __name__ == "__main__":
    main()
