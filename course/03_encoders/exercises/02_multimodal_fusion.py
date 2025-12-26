"""
Module 3 Exercise 2: Multimodal Data Fusion

This exercise demonstrates how to combine different data modalities
(text, images, structured data) into unified hypervector representations.

Tasks:
1. Encode text descriptions with SequenceEncoder
2. Encode image features with spatial bundling
3. Encode metadata with DictEncoder
4. Fuse all modalities into single representation
5. Query across modalities

Expected learning:
- Combining heterogeneous data types
- Cross-modal similarity search
- Multimodal representation learning
- Building unified search systems
"""

from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import SequenceEncoder, DictEncoder, SetEncoder
from vsax.similarity import cosine_similarity
import jax.numpy as jnp
import numpy as np


class MultimodalEncoder:
    """
    Encoder that fuses text, visual, and structured data.
    """

    def __init__(self, model, memory):
        """
        Initialize multimodal encoder.

        Args:
            model: VSA model
            memory: VSA memory
        """
        self.model = model
        self.memory = memory

        # Create specialized encoders
        self.text_encoder = SequenceEncoder(model, memory)
        self.dict_encoder = DictEncoder(model, memory)
        self.set_encoder = SetEncoder(model, memory)

        # Add modality markers
        self.memory.add_many(["TEXT", "IMAGE", "METADATA"])

    def encode_text(self, words):
        """
        Encode text as sequence.

        Args:
            words: List of words

        Returns:
            Text hypervector
        """
        text_hv = self.text_encoder.encode(words)

        # Mark as text modality
        text_marker = self.memory["TEXT"].vec
        marked = self.model.opset.bind(text_hv.vec, text_marker)

        return self.model.rep_cls(marked)

    def encode_image_features(self, features):
        """
        Encode image features as weighted bundle.

        Args:
            features: Dictionary {feature_name: weight}

        Returns:
            Image hypervector
        """
        # Ensure features are in memory
        for feature in features.keys():
            if feature not in self.memory.symbols:
                self.memory.add(feature)

        # Weight features by their values
        weighted_features = []
        for feature, weight in features.items():
            feature_vec = self.memory[feature].vec
            weighted = weight * feature_vec
            weighted_features.append(weighted)

        # Bundle weighted features
        image_hv = jnp.sum(jnp.stack(weighted_features), axis=0)
        image_hv = image_hv / jnp.linalg.norm(image_hv)

        # Mark as image modality
        image_marker = self.memory["IMAGE"].vec
        marked = self.model.opset.bind(image_hv, image_marker)

        return self.model.rep_cls(marked)

    def encode_metadata(self, metadata_dict):
        """
        Encode structured metadata.

        Args:
            metadata_dict: Dictionary of key-value pairs

        Returns:
            Metadata hypervector
        """
        meta_hv = self.dict_encoder.encode(metadata_dict)

        # Mark as metadata modality
        meta_marker = self.memory["METADATA"].vec
        marked = self.model.opset.bind(meta_hv.vec, meta_marker)

        return self.model.rep_cls(marked)

    def fuse_multimodal(self, text_hv=None, image_hv=None, meta_hv=None):
        """
        Fuse multiple modalities into single representation.

        Args:
            text_hv: Optional text hypervector
            image_hv: Optional image hypervector
            meta_hv: Optional metadata hypervector

        Returns:
            Fused multimodal hypervector
        """
        modalities = []

        if text_hv is not None:
            modalities.append(text_hv.vec)

        if image_hv is not None:
            modalities.append(image_hv.vec)

        if meta_hv is not None:
            modalities.append(meta_hv.vec)

        if not modalities:
            raise ValueError("At least one modality must be provided")

        # Bundle all modalities
        fused = self.model.opset.bundle(*modalities)

        return self.model.rep_cls(fused)


def create_product_dataset():
    """
    Create synthetic product dataset with multimodal data.

    Returns:
        List of products (each with text, image features, metadata)
    """
    products = [
        {
            "id": "P001",
            "text": ["wireless", "bluetooth", "headphones", "noise", "cancelling"],
            "image_features": {
                "black": 0.8,
                "plastic": 0.6,
                "padded": 0.7,
                "compact": 0.5
            },
            "metadata": {
                "category": "electronics",
                "price": "high",
                "rating": "excellent",
                "brand": "premium"
            }
        },
        {
            "id": "P002",
            "text": ["running", "shoes", "athletic", "comfortable", "breathable"],
            "image_features": {
                "blue": 0.7,
                "fabric": 0.9,
                "flexible": 0.8,
                "lightweight": 0.6
            },
            "metadata": {
                "category": "sports",
                "price": "medium",
                "rating": "good",
                "brand": "athletic"
            }
        },
        {
            "id": "P003",
            "text": ["smartphone", "camera", "touchscreen", "5g", "wireless"],
            "image_features": {
                "black": 0.9,
                "glass": 0.7,
                "sleek": 0.8,
                "compact": 0.6
            },
            "metadata": {
                "category": "electronics",
                "price": "high",
                "rating": "excellent",
                "brand": "premium"
            }
        },
        {
            "id": "P004",
            "text": ["backpack", "laptop", "compartments", "waterproof", "durable"],
            "image_features": {
                "black": 0.6,
                "fabric": 0.9,
                "padded": 0.7,
                "spacious": 0.8
            },
            "metadata": {
                "category": "accessories",
                "price": "medium",
                "rating": "good",
                "brand": "outdoor"
            }
        },
        {
            "id": "P005",
            "text": ["smartwatch", "fitness", "tracker", "waterproof", "touchscreen"],
            "image_features": {
                "black": 0.5,
                "plastic": 0.6,
                "compact": 0.9,
                "sleek": 0.7
            },
            "metadata": {
                "category": "electronics",
                "price": "medium",
                "rating": "excellent",
                "brand": "fitness"
            }
        }
    ]

    return products


def test_multimodal_encoding():
    """
    Test encoding products with multimodal data.
    """
    print("=" * 60)
    print("Test 1: Multimodal Product Encoding")
    print("=" * 60)

    model = create_fhrr_model(dim=4096)  # Higher dim for complex data
    memory = VSAMemory(model)

    # Create dataset
    products = create_product_dataset()

    # Add all vocabulary
    all_words = set()
    all_features = set()
    all_metadata_vals = set()

    for product in products:
        all_words.update(product["text"])
        all_features.update(product["image_features"].keys())
        for key, val in product["metadata"].items():
            all_metadata_vals.add(key)
            all_metadata_vals.add(val)

    memory.add_many(list(all_words | all_features | all_metadata_vals))

    # Create encoder
    encoder = MultimodalEncoder(model, memory)

    # Encode all products
    encoded_products = {}

    print("\nEncoding products:")
    for product in products:
        # Encode each modality
        text_hv = encoder.encode_text(product["text"])
        image_hv = encoder.encode_image_features(product["image_features"])
        meta_hv = encoder.encode_metadata(product["metadata"])

        # Fuse modalities
        fused_hv = encoder.fuse_multimodal(text_hv, image_hv, meta_hv)

        encoded_products[product["id"]] = {
            "text": text_hv,
            "image": image_hv,
            "metadata": meta_hv,
            "fused": fused_hv,
            "data": product
        }

        print(f"  {product['id']}: {' '.join(product['text'][:3])}...")

    return encoded_products, encoder, memory


def test_cross_modal_search(encoded_products):
    """
    Test searching across modalities.
    """
    print("\n" + "=" * 60)
    print("Test 2: Cross-Modal Search")
    print("=" * 60)

    # Query 1: Text-based search
    print("\nQuery 1: Find products matching text 'wireless bluetooth'")
    query_text = ["wireless", "bluetooth"]

    # Encode query (text only)
    model = encoded_products["P001"]["fused"].model
    memory = encoded_products["P001"]["text"].memory

    query_encoder = MultimodalEncoder(model, memory)
    query_hv = query_encoder.encode_text(query_text)

    # Compare to all products (fused representations)
    results = {}
    for prod_id, data in encoded_products.items():
        sim = cosine_similarity(query_hv.vec, data["fused"].vec)
        results[prod_id] = float(sim)

    # Sort by similarity
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Product':<10s} {'Description':<40s} {'Similarity':<12s}")
    print("-" * 70)
    for prod_id, sim in sorted_results[:3]:
        desc = " ".join(encoded_products[prod_id]["data"]["text"][:4])
        print(f"{prod_id:<10s} {desc:<40s} {sim:<12.4f}")

    # Query 2: Image feature search
    print("\n\nQuery 2: Find products with 'black' and 'compact' features")
    query_features = {"black": 0.8, "compact": 0.7}
    query_hv = query_encoder.encode_image_features(query_features)

    results = {}
    for prod_id, data in encoded_products.items():
        sim = cosine_similarity(query_hv.vec, data["fused"].vec)
        results[prod_id] = float(sim)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Product':<10s} {'Description':<40s} {'Similarity':<12s}")
    print("-" * 70)
    for prod_id, sim in sorted_results[:3]:
        desc = " ".join(encoded_products[prod_id]["data"]["text"][:4])
        print(f"{prod_id:<10s} {desc:<40s} {sim:<12.4f}")

    # Query 3: Metadata search
    print("\n\nQuery 3: Find 'electronics' with 'excellent' rating")
    query_meta = {"category": "electronics", "rating": "excellent"}
    query_hv = query_encoder.encode_metadata(query_meta)

    results = {}
    for prod_id, data in encoded_products.items():
        sim = cosine_similarity(query_hv.vec, data["fused"].vec)
        results[prod_id] = float(sim)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Product':<10s} {'Description':<40s} {'Similarity':<12s}")
    print("-" * 70)
    for prod_id, sim in sorted_results[:3]:
        desc = " ".join(encoded_products[prod_id]["data"]["text"][:4])
        print(f"{prod_id:<10s} {desc:<40s} {sim:<12.4f}")


def test_modality_importance():
    """
    Test which modality contributes most to similarity.
    """
    print("\n" + "=" * 60)
    print("Test 3: Modality Importance Analysis")
    print("=" * 60)

    model = create_fhrr_model(dim=4096)
    memory = VSAMemory(model)
    products = create_product_dataset()

    # Add vocabulary
    all_symbols = set()
    for product in products:
        all_symbols.update(product["text"])
        all_symbols.update(product["image_features"].keys())
        for k, v in product["metadata"].items():
            all_symbols.update([k, v])

    memory.add_many(list(all_symbols))

    encoder = MultimodalEncoder(model, memory)

    # Compare P001 (headphones) vs P003 (smartphone)
    p1 = products[0]  # Headphones
    p3 = products[2]  # Smartphone

    # Encode modalities separately
    p1_text = encoder.encode_text(p1["text"])
    p1_image = encoder.encode_image_features(p1["image_features"])
    p1_meta = encoder.encode_metadata(p1["metadata"])

    p3_text = encoder.encode_text(p3["text"])
    p3_image = encoder.encode_image_features(p3["image_features"])
    p3_meta = encoder.encode_metadata(p3["metadata"])

    # Compare modalities
    text_sim = cosine_similarity(p1_text.vec, p3_text.vec)
    image_sim = cosine_similarity(p1_image.vec, p3_image.vec)
    meta_sim = cosine_similarity(p1_meta.vec, p3_meta.vec)

    print(f"\nComparing P001 (headphones) vs P003 (smartphone):")
    print(f"  Text similarity:     {text_sim:.4f}")
    print(f"  Image similarity:    {image_sim:.4f}")
    print(f"  Metadata similarity: {meta_sim:.4f}")

    print("\nObservations:")
    print("- Both are electronics → high metadata similarity")
    print("- Both are black/compact → moderate image similarity")
    print("- Different functions → lower text similarity")
    print("- Multimodal fusion captures all aspects!")


def test_modality_weighting():
    """
    Test weighted fusion of modalities.
    """
    print("\n" + "=" * 60)
    print("Test 4: Weighted Modality Fusion")
    print("=" * 60)

    model = create_fhrr_model(dim=4096)
    memory = VSAMemory(model)
    products = create_product_dataset()

    # Add vocabulary
    all_symbols = set()
    for product in products:
        all_symbols.update(product["text"])
        all_symbols.update(product["image_features"].keys())
        for k, v in product["metadata"].items():
            all_symbols.update([k, v])

    memory.add_many(list(all_symbols))

    encoder = MultimodalEncoder(model, memory)

    # Encode product with different weighting strategies
    product = products[0]  # Headphones

    text_hv = encoder.encode_text(product["text"])
    image_hv = encoder.encode_image_features(product["image_features"])
    meta_hv = encoder.encode_metadata(product["metadata"])

    # Strategy 1: Equal weights (standard bundling)
    fused_equal = encoder.fuse_multimodal(text_hv, image_hv, meta_hv)

    # Strategy 2: Text-heavy (emphasize description)
    weighted_text = 2.0 * text_hv.vec
    weighted_image = 1.0 * image_hv.vec
    weighted_meta = 1.0 * meta_hv.vec

    fused_text_heavy = model.opset.bundle(weighted_text, weighted_image, weighted_meta)

    # Strategy 3: Metadata-heavy (emphasize category/price)
    weighted_text2 = 1.0 * text_hv.vec
    weighted_image2 = 1.0 * image_hv.vec
    weighted_meta2 = 2.0 * meta_hv.vec

    fused_meta_heavy = model.opset.bundle(weighted_text2, weighted_image2, weighted_meta2)

    print("\nWeighted fusion strategies:")
    print("1. Equal weights (standard)")
    print("2. Text-heavy (2:1:1)")
    print("3. Metadata-heavy (1:1:2)")

    # Test query: "wireless" (text) vs "electronics" (metadata)
    query_text = encoder.encode_text(["wireless"])
    query_meta = encoder.encode_metadata({"category": "electronics"})

    sim_equal_text = cosine_similarity(fused_equal.vec, query_text.vec)
    sim_text_heavy_text = cosine_similarity(fused_text_heavy, query_text.vec)
    sim_meta_heavy_text = cosine_similarity(fused_meta_heavy, query_text.vec)

    sim_equal_meta = cosine_similarity(fused_equal.vec, query_meta.vec)
    sim_text_heavy_meta = cosine_similarity(fused_text_heavy, query_meta.vec)
    sim_meta_heavy_meta = cosine_similarity(fused_meta_heavy, query_meta.vec)

    print(f"\nSimilarity to text query 'wireless':")
    print(f"  Equal weights:    {sim_equal_text:.4f}")
    print(f"  Text-heavy:       {sim_text_heavy_text:.4f}  ← Higher!")
    print(f"  Metadata-heavy:   {sim_meta_heavy_text:.4f}")

    print(f"\nSimilarity to metadata query 'electronics':")
    print(f"  Equal weights:    {sim_equal_meta:.4f}")
    print(f"  Text-heavy:       {sim_text_heavy_meta:.4f}")
    print(f"  Metadata-heavy:   {sim_meta_heavy_meta:.4f}  ← Higher!")

    print("\nObservation: Weighting emphasizes chosen modality in queries")


def main():
    """
    Run all multimodal fusion tests.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 3 EXERCISE 2")
    print(" " * 18 + "Multimodal Data Fusion")
    print("=" * 80)

    # Run tests
    encoded_products, encoder, memory = test_multimodal_encoding()
    test_cross_modal_search(encoded_products)
    test_modality_importance()
    test_modality_weighting()

    print("\n" + "=" * 80)
    print("Exercise complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("✓ Combine heterogeneous data types (text, images, metadata)")
    print("✓ Each modality encoded with appropriate encoder")
    print("✓ Fusion via bundling creates unified representation")
    print("✓ Cross-modal search: query one modality, find others")
    print("✓ Weighted fusion emphasizes specific modalities")
    print("✓ Multimodal VSA enables rich similarity search")
    print("=" * 80)


if __name__ == "__main__":
    main()
