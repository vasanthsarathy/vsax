"""Vector Function Architecture - Image Processing Example.

This example demonstrates image encoding and manipulation using VFA, reproducing
concepts from Frady et al. 2021 §7.1.

Note: Current implementation uses simplified 1D flattened encoding. Full 2D
encoding with multi-dimensional VFA is planned for future work.

Based on:
    Frady et al. 2021: "Computing on Functions Using Randomized Vector
    Representations"
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from vsax import VSAMemory, create_fhrr_model
from vsax.vfa import ImageProcessor


def create_simple_images():
    """Create simple test images."""
    # Image 1: Vertical stripes
    img1 = jnp.zeros((16, 16))
    img1 = img1.at[:, 0:4].set(1.0)
    img1 = img1.at[:, 8:12].set(1.0)

    # Image 2: Horizontal stripes
    img2 = jnp.zeros((16, 16))
    img2 = img2.at[0:4, :].set(1.0)
    img2 = img2.at[8:12, :].set(1.0)

    # Image 3: Checkerboard
    img3 = jnp.zeros((16, 16))
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            if (i + j) % 8 == 0:
                img3 = img3.at[i : i + 4, j : j + 4].set(1.0)

    # Image 4: Circle
    img4 = jnp.zeros((16, 16))
    center = 7.5
    radius = 5.0
    for i in range(16):
        for j in range(16):
            if (i - center) ** 2 + (j - center) ** 2 <= radius**2:
                img4 = img4.at[i, j].set(1.0)

    return img1, img2, img3, img4


def main():
    """Demonstrate image processing with VFA."""
    print("=" * 70)
    print("Vector Function Architecture - Image Processing")
    print("=" * 70)

    # Initialize FHRR model with high dimensionality for images
    key = jax.random.PRNGKey(42)
    model = create_fhrr_model(dim=4096, key=key)
    memory = VSAMemory(model)

    print("\n1. Creating Test Images")
    print("-" * 70)

    img1, img2, img3, img4 = create_simple_images()
    images = {
        "Vertical Stripes": img1,
        "Horizontal Stripes": img2,
        "Checkerboard": img3,
        "Circle": img4,
    }

    print(f"  Created {len(images)} test images (16x16 pixels)")
    for name in images.keys():
        print(f"    - {name}")

    print("\n2. Encoding Images as Hypervectors")
    print("-" * 70)

    processor = ImageProcessor(model, memory)
    encoded_images = {}

    for name, img in images.items():
        img_hv = processor.encode(img)
        encoded_images[name] = img_hv
        print(f"  {name:20s}: shape {img.shape} -> hypervector dim {img_hv.shape[0]}")

    print("\n3. Decoding Images")
    print("-" * 70)

    decoded_images = {}
    reconstruction_errors = {}

    for name, img_hv in encoded_images.items():
        decoded = processor.decode(img_hv, shape=(16, 16))
        decoded_images[name] = decoded

        # Compute reconstruction error
        original = images[name]
        mse = jnp.mean((decoded - original) ** 2)
        mae = jnp.mean(jnp.abs(decoded - original))
        reconstruction_errors[name] = (mse, mae)

        print(f"  {name:20s}: MSE = {mse:.6f}, MAE = {mae:.6f}")

    print("\n4. Image Blending")
    print("-" * 70)

    # Blend different pairs of images
    blends = [
        ("Vertical Stripes", "Horizontal Stripes", 0.5),
        ("Checkerboard", "Circle", 0.3),
        ("Vertical Stripes", "Circle", 0.7),
    ]

    blended_images = {}

    for name1, name2, alpha in blends:
        hv1 = encoded_images[name1]
        hv2 = encoded_images[name2]

        blended_hv = processor.blend(hv1, hv2, alpha=alpha)
        blended_img = processor.decode(blended_hv, shape=(16, 16))
        blended_images[f"{name1[:8]}+{name2[:8]}"] = blended_img

        print(f"  Blended '{name1}' and '{name2}' (alpha={alpha})")

    print("\n5. Visualization")
    print("-" * 70)

    # Plot original images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, (name, img) in enumerate(images.items()):
        ax = axes[0, idx]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Original: {name}", fontsize=9)
        ax.axis("off")

    # Plot reconstructed images
    for idx, (name, img) in enumerate(decoded_images.items()):
        ax = axes[1, idx]
        ax.imshow(img, cmap="gray")
        mse, mae = reconstruction_errors[name]
        ax.set_title(f"Reconstructed\nMSE={mse:.4f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("vfa_image_reconstruction.png", dpi=150, bbox_inches="tight")
    print("  [OK] Reconstruction plots saved to vfa_image_reconstruction.png")

    # Plot blended images
    fig, axes = plt.subplots(1, len(blends) + 1, figsize=(12, 3))

    for idx, ((name1, name2, alpha), (blend_name, blended_img)) in enumerate(
        zip(blends, blended_images.items())
    ):
        ax = axes[idx]
        ax.imshow(blended_img, cmap="gray")
        ax.set_title(f"{name1[:8]} +\n{name2[:8]}\n(alpha={alpha})", fontsize=8)
        ax.axis("off")

    # Show original for reference
    ax = axes[-1]
    ax.imshow(img1, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Reference:\nVertical", fontsize=8)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("vfa_image_blending.png", dpi=150, bbox_inches="tight")
    print("  [OK] Blending plots saved to vfa_image_blending.png")

    print("\n6. Image Similarity")
    print("-" * 70)

    from vsax.similarity import cosine_similarity

    # Compute pairwise similarities
    image_names = list(images.keys())

    print("  Pairwise image similarities:\n")
    print("  " + " " * 20, end="")
    for name in image_names:
        print(f"{name[:8]:>10s}", end="")
    print()

    for name1 in image_names:
        print(f"  {name1:20s}", end="")
        for name2 in image_names:
            hv1 = encoded_images[name1]
            hv2 = encoded_images[name2]
            sim = cosine_similarity(hv1.vec, hv2.vec)
            print(f"{sim:10.4f}", end="")
        print()

    print("\n7. Comparison: Different Image Sizes")
    print("-" * 70)

    # Test with different image sizes
    sizes = [(8, 8), (16, 16), (32, 32)]
    size_results = {}

    for size in sizes:
        # Create simple test image
        test_img = jnp.zeros(size)
        h, w = size
        test_img = test_img.at[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].set(1.0)

        # Encode and decode
        processor_size = ImageProcessor(model, VSAMemory(model))
        encoded = processor_size.encode(test_img)
        decoded = processor_size.decode(encoded, shape=size)

        # Compute error
        mse = jnp.mean((decoded - test_img) ** 2)
        size_results[size] = mse

        print(f"  Size {size[0]:2d}x{size[1]:2d}: MSE = {mse:.6f}")

    print("\n8. Limitations")
    print("-" * 70)
    print("  NOTE: Current implementation uses simplified 1D flattened encoding.")
    print("  True 2D spatial operations (shifting, rotation) require")
    print("  multi-dimensional VFA, planned for future work.")
    print()
    print("  For now, VFA can:")
    print("    [OK] Encode images as compact hypervectors")
    print("    [OK] Decode hypervectors back to images")
    print("    [OK] Blend multiple images")
    print("    [OK] Compute image similarities")
    print()
    print("  Future work:")
    print("    - Full 2D VFA for spatial transformations")
    print("    - Image shifting and rotation")
    print("    - Multi-scale representations")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Images can be encoded as hypervectors using VFA")
    print("  - Reconstruction quality depends on dimensionality")
    print("  - Image blending is natural in hypervector space")
    print("  - Compact representation enables efficient storage/retrieval")
    print("=" * 70)


if __name__ == "__main__":
    main()
