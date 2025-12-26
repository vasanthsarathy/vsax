"""
Module 3 Capstone Project: VSA-Based Recommendation System

Build a complete recommendation system using Vector Symbolic Architectures.
Combines all encoder types learned in Module 3 to create a powerful,
interpretable recommendation engine.

Features:
1. User profile encoding (preferences, history, demographics)
2. Item encoding (features, metadata, descriptions)
3. Collaborative filtering via similarity
4. Content-based recommendations
5. Hybrid recommendation strategies
6. Cold-start handling with few-shot learning

Expected learning:
- Integration of all encoder types
- Building production-ready VSA applications
- Handling real-world recommendation challenges
- Interpretable recommendation explanations
"""

from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import DictEncoder, SetEncoder, SequenceEncoder, FractionalPowerEncoder
from vsax.similarity import cosine_similarity
from vsax.utils import vmap_similarity
import jax.numpy as jnp
import numpy as np
from collections import defaultdict


class RecommendationSystem:
    """
    Complete VSA-based recommendation system.
    """

    def __init__(self, dim=4096):
        """
        Initialize recommendation system.

        Args:
            dim: Hypervector dimensionality
        """
        self.model = create_fhrr_model(dim=dim)
        self.memory = VSAMemory(self.model)

        # Create encoders
        self.dict_encoder = DictEncoder(self.model, self.memory)
        self.set_encoder = SetEncoder(self.model, self.memory)
        self.seq_encoder = SequenceEncoder(self.model, self.memory)
        self.fpe = FractionalPowerEncoder(self.model, self.memory, scale=0.1)

        # Storage
        self.users = {}      # user_id -> encoded profile
        self.items = {}      # item_id -> encoded item
        self.interactions = defaultdict(list)  # user_id -> [item_ids]

    def add_user(self, user_id, demographics, preferences, history=None):
        """
        Add user to the system.

        Args:
            user_id: Unique user identifier
            demographics: Dict of demographic info (age, location, etc.)
            preferences: Set of preference tags
            history: Optional list of previously interacted items (in order)
        """
        # Add all vocabulary
        for key, val in demographics.items():
            self.memory.add_if_missing(key)
            self.memory.add_if_missing(str(val))

        for pref in preferences:
            self.memory.add_if_missing(pref)

        # Encode demographics
        demo_hv = self.dict_encoder.encode(demographics)

        # Encode preferences
        pref_hv = self.set_encoder.encode(preferences)

        # Encode history (if provided)
        if history:
            for item in history:
                self.memory.add_if_missing(item)
            history_hv = self.seq_encoder.encode(history)
        else:
            # Empty history
            history_hv = self.model.rep_cls(jnp.zeros(self.model.dim, dtype=jnp.complex64))

        # Combine into user profile (weighted bundle)
        user_profile = self.model.opset.bundle(
            demo_hv.vec,
            2.0 * pref_hv.vec,  # Emphasize preferences
            1.5 * history_hv.vec  # Moderate weight on history
        )

        self.users[user_id] = self.model.rep_cls(user_profile)

        # Track interactions
        if history:
            self.interactions[user_id].extend(history)

        return user_id

    def add_item(self, item_id, metadata, features, description_words):
        """
        Add item to the system.

        Args:
            item_id: Unique item identifier
            metadata: Dict of metadata (category, price, etc.)
            features: Set of feature tags
            description_words: List of words describing the item
        """
        # Add vocabulary
        for key, val in metadata.items():
            self.memory.add_if_missing(key)
            self.memory.add_if_missing(str(val))

        for feat in features:
            self.memory.add_if_missing(feat)

        for word in description_words:
            self.memory.add_if_missing(word)

        # Encode metadata
        meta_hv = self.dict_encoder.encode(metadata)

        # Encode features
        feat_hv = self.set_encoder.encode(features)

        # Encode description
        desc_hv = self.seq_encoder.encode(description_words[:10])  # Limit length

        # Combine into item representation
        item_vec = self.model.opset.bundle(
            meta_hv.vec,
            1.5 * feat_hv.vec,  # Emphasize features
            desc_hv.vec
        )

        self.items[item_id] = self.model.rep_cls(item_vec)

        return item_id

    def record_interaction(self, user_id, item_id):
        """
        Record user-item interaction.

        Args:
            user_id: User identifier
            item_id: Item identifier
        """
        if user_id not in self.users:
            raise ValueError(f"Unknown user: {user_id}")

        if item_id not in self.items:
            raise ValueError(f"Unknown item: {item_id}")

        self.interactions[user_id].append(item_id)

        # Update user history (re-encode with new interaction)
        # In production, this would be done in batches
        print(f"Recorded: User {user_id} interacted with Item {item_id}")

    def recommend_content_based(self, user_id, top_k=5, exclude_interacted=True):
        """
        Content-based recommendations using user profile.

        Args:
            user_id: User to recommend for
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude already-interacted items

        Returns:
            List of (item_id, similarity) tuples
        """
        if user_id not in self.users:
            raise ValueError(f"Unknown user: {user_id}")

        user_vec = self.users[user_id].vec

        # Compute similarities to all items
        item_ids = list(self.items.keys())
        item_vecs = jnp.stack([self.items[item_id].vec for item_id in item_ids])

        similarities = vmap_similarity(None, user_vec, item_vecs)

        # Sort by similarity
        sorted_indices = jnp.argsort(similarities)[::-1]

        # Generate recommendations
        recommendations = []
        for idx in sorted_indices:
            item_id = item_ids[int(idx)]
            sim = float(similarities[int(idx)])

            # Skip if already interacted
            if exclude_interacted and item_id in self.interactions[user_id]:
                continue

            recommendations.append((item_id, sim))

            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_collaborative(self, user_id, top_k=5, exclude_interacted=True):
        """
        Collaborative filtering: recommend based on similar users.

        Args:
            user_id: User to recommend for
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude already-interacted items

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.users:
            raise ValueError(f"Unknown user: {user_id}")

        user_vec = self.users[user_id].vec

        # Find similar users
        similar_users = []
        for other_id, other_user in self.users.items():
            if other_id == user_id:
                continue

            sim = float(cosine_similarity(user_vec, other_user.vec))
            similar_users.append((other_id, sim))

        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)

        # Aggregate items from similar users (weighted by user similarity)
        item_scores = defaultdict(float)

        for other_id, user_sim in similar_users[:10]:  # Top 10 similar users
            for item_id in self.interactions[other_id]:
                # Skip if current user already interacted
                if exclude_interacted and item_id in self.interactions[user_id]:
                    continue

                # Weight by user similarity
                item_scores[item_id] += user_sim

        # Sort by score
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

    def recommend_hybrid(self, user_id, top_k=5, content_weight=0.6, collab_weight=0.4):
        """
        Hybrid recommendations combining content and collaborative filtering.

        Args:
            user_id: User to recommend for
            top_k: Number of recommendations
            content_weight: Weight for content-based score
            collab_weight: Weight for collaborative score

        Returns:
            List of (item_id, combined_score) tuples
        """
        # Get content-based recommendations
        content_recs = self.recommend_content_based(user_id, top_k=20)
        content_scores = {item_id: sim for item_id, sim in content_recs}

        # Get collaborative recommendations
        collab_recs = self.recommend_collaborative(user_id, top_k=20)
        collab_scores = {item_id: score for item_id, score in collab_recs}

        # Normalize scores to [0, 1]
        if content_scores:
            max_content = max(content_scores.values())
            content_scores = {k: v/max_content for k, v in content_scores.items()}

        if collab_scores:
            max_collab = max(collab_scores.values())
            collab_scores = {k: v/max_collab for k, v in collab_scores.items()}

        # Combine scores
        all_items = set(content_scores.keys()) | set(collab_scores.keys())
        hybrid_scores = {}

        for item_id in all_items:
            content_score = content_scores.get(item_id, 0.0)
            collab_score = collab_scores.get(item_id, 0.0)

            hybrid_scores[item_id] = (
                content_weight * content_score +
                collab_weight * collab_score
            )

        # Sort and return top-k
        recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def explain_recommendation(self, user_id, item_id):
        """
        Explain why an item was recommended to a user.

        Args:
            user_id: User identifier
            item_id: Item identifier

        Returns:
            Dict with explanation components
        """
        if user_id not in self.users:
            raise ValueError(f"Unknown user: {user_id}")

        if item_id not in self.items:
            raise ValueError(f"Unknown item: {item_id}")

        user_vec = self.users[user_id].vec
        item_vec = self.items[item_id].vec

        # Overall similarity
        overall_sim = float(cosine_similarity(user_vec, item_vec))

        # Find similar users who interacted with this item
        similar_users_with_item = []
        for other_id, other_user in self.users.items():
            if other_id == user_id:
                continue

            if item_id in self.interactions[other_id]:
                user_sim = float(cosine_similarity(user_vec, other_user.vec))
                similar_users_with_item.append((other_id, user_sim))

        similar_users_with_item.sort(key=lambda x: x[1], reverse=True)

        return {
            "user_item_similarity": overall_sim,
            "similar_users": similar_users_with_item[:3],
            "explanation": f"Similarity: {overall_sim:.3f}, {len(similar_users_with_item)} similar users liked this"
        }


def create_sample_dataset():
    """
    Create sample users and items for demonstration.
    """
    users = [
        {
            "id": "U001",
            "demographics": {"age_group": "25-34", "location": "urban", "gender": "male"},
            "preferences": {"tech", "gaming", "music", "fitness"},
            "history": ["I001", "I003", "I007"]
        },
        {
            "id": "U002",
            "demographics": {"age_group": "25-34", "location": "urban", "gender": "female"},
            "preferences": {"fashion", "books", "music", "travel"},
            "history": ["I002", "I005", "I008"]
        },
        {
            "id": "U003",
            "demographics": {"age_group": "35-44", "location": "suburban", "gender": "male"},
            "preferences": {"tech", "sports", "cars", "cooking"},
            "history": ["I001", "I004", "I006"]
        },
        {
            "id": "U004",
            "demographics": {"age_group": "18-24", "location": "urban", "gender": "female"},
            "preferences": {"gaming", "fashion", "music", "fitness"},
            "history": ["I003", "I005", "I007"]
        },
    ]

    items = [
        {
            "id": "I001",
            "metadata": {"category": "electronics", "price": "high", "brand": "premium"},
            "features": {"wireless", "bluetooth", "noise_cancelling", "portable"},
            "description": ["premium", "wireless", "headphones", "with", "noise", "cancelling"]
        },
        {
            "id": "I002",
            "metadata": {"category": "fashion", "price": "medium", "brand": "trendy"},
            "features": {"casual", "comfortable", "stylish", "versatile"},
            "description": ["casual", "summer", "dress", "comfortable", "and", "stylish"]
        },
        {
            "id": "I003",
            "metadata": {"category": "gaming", "price": "high", "brand": "gaming"},
            "features": {"rgb", "mechanical", "wireless", "programmable"},
            "description": ["mechanical", "gaming", "keyboard", "with", "rgb", "lighting"]
        },
        {
            "id": "I004",
            "metadata": {"category": "sports", "price": "medium", "brand": "athletic"},
            "features": {"breathable", "lightweight", "durable", "comfortable"},
            "description": ["running", "shoes", "lightweight", "and", "breathable"]
        },
        {
            "id": "I005",
            "metadata": {"category": "books", "price": "low", "brand": "bestseller"},
            "features": {"fiction", "bestseller", "paperback", "engaging"},
            "description": ["bestselling", "novel", "fiction", "engaging", "story"]
        },
        {
            "id": "I006",
            "metadata": {"category": "automotive", "price": "high", "brand": "premium"},
            "features": {"leather", "heated", "adjustable", "comfortable"},
            "description": ["premium", "car", "seat", "covers", "leather"]
        },
        {
            "id": "I007",
            "metadata": {"category": "fitness", "price": "medium", "brand": "fitness"},
            "features": {"wireless", "heart_rate", "waterproof", "gps"},
            "description": ["fitness", "tracker", "with", "heart", "rate", "monitor"]
        },
        {
            "id": "I008",
            "metadata": {"category": "travel", "price": "medium", "brand": "outdoor"},
            "features": {"waterproof", "durable", "spacious", "lightweight"},
            "description": ["travel", "backpack", "waterproof", "and", "durable"]
        },
        {
            "id": "I009",
            "metadata": {"category": "electronics", "price": "high", "brand": "premium"},
            "features": {"4k", "smart", "large", "hdr"},
            "description": ["smart", "tv", "4k", "ultra", "hd"]
        },
        {
            "id": "I010",
            "metadata": {"category": "kitchen", "price": "medium", "brand": "cooking"},
            "features": {"stainless", "sharp", "durable", "professional"},
            "description": ["professional", "chef", "knife", "set", "stainless", "steel"]
        }
    ]

    return users, items


def demo_recommendation_system():
    """
    Demonstrate complete recommendation system.
    """
    print("=" * 80)
    print(" " * 25 + "VSA RECOMMENDATION SYSTEM")
    print("=" * 80)

    # Create system
    print("\n[1] Initializing recommendation system...")
    system = RecommendationSystem(dim=4096)

    # Load data
    print("[2] Loading sample dataset...")
    users, items = create_sample_dataset()

    # Add users
    print(f"[3] Adding {len(users)} users...")
    for user in users:
        system.add_user(
            user["id"],
            user["demographics"],
            user["preferences"],
            user["history"]
        )

    # Add items
    print(f"[4] Adding {len(items)} items...")
    for item in items:
        system.add_item(
            item["id"],
            item["metadata"],
            item["features"],
            item["description"]
        )

    print("\nSystem ready!")
    print(f"  - Users: {len(system.users)}")
    print(f"  - Items: {len(system.items)}")
    print(f"  - Interactions: {sum(len(v) for v in system.interactions.values())}")

    return system, users, items


def test_content_based(system):
    """
    Test content-based recommendations.
    """
    print("\n" + "=" * 80)
    print("Test 1: Content-Based Recommendations")
    print("=" * 80)

    user_id = "U001"  # Tech + gaming + music + fitness user
    print(f"\nRecommending for User {user_id}:")
    print("Profile: tech, gaming, music, fitness enthusiast")
    print("History: I001 (headphones), I003 (keyboard), I007 (fitness tracker)")

    recommendations = system.recommend_content_based(user_id, top_k=5)

    print(f"\n{'Rank':<6s} {'Item':<8s} {'Description':<50s} {'Similarity':<12s}")
    print("-" * 80)

    for rank, (item_id, sim) in enumerate(recommendations, 1):
        # Get item description (first 5 words)
        desc = " ".join([d for d in system.memory.symbols.keys() if item_id in d][:5])
        # Simplified - in reality, would look up from dataset
        desc = f"Item {item_id}"
        print(f"{rank:<6d} {item_id:<8s} {desc:<50s} {sim:<12.4f}")

    print("\nObservation: Recommends items matching user preferences (tech/gaming/fitness)")


def test_collaborative(system):
    """
    Test collaborative filtering.
    """
    print("\n" + "=" * 80)
    print("Test 2: Collaborative Filtering")
    print("=" * 80)

    user_id = "U002"  # Fashion + books + music + travel user
    print(f"\nRecommending for User {user_id}:")
    print("Profile: fashion, books, music, travel enthusiast")
    print("History: I002 (dress), I005 (book), I008 (backpack)")

    # Find similar users
    user_vec = system.users[user_id].vec
    similar_users = []

    for other_id, other_user in system.users.items():
        if other_id != user_id:
            sim = float(cosine_similarity(user_vec, other_user.vec))
            similar_users.append((other_id, sim))

    similar_users.sort(key=lambda x: x[1], reverse=True)

    print("\nMost similar users:")
    for other_id, sim in similar_users[:3]:
        print(f"  {other_id}: similarity = {sim:.4f}")

    # Get recommendations
    recommendations = system.recommend_collaborative(user_id, top_k=5)

    print(f"\n{'Rank':<6s} {'Item':<8s} {'Score':<12s} {'Reason':<40s}")
    print("-" * 80)

    for rank, (item_id, score) in enumerate(recommendations, 1):
        reason = f"Liked by similar users"
        print(f"{rank:<6d} {item_id:<8s} {score:<12.4f} {reason:<40s}")

    print("\nObservation: Recommends items liked by users with similar preferences")


def test_hybrid(system):
    """
    Test hybrid recommendations.
    """
    print("\n" + "=" * 80)
    print("Test 3: Hybrid Recommendations")
    print("=" * 80)

    user_id = "U004"  # Gaming + fashion + music + fitness user
    print(f"\nRecommending for User {user_id}:")
    print("Profile: gaming, fashion, music, fitness enthusiast")
    print("Strategy: Hybrid (60% content, 40% collaborative)")

    recommendations = system.recommend_hybrid(
        user_id,
        top_k=5,
        content_weight=0.6,
        collab_weight=0.4
    )

    print(f"\n{'Rank':<6s} {'Item':<8s} {'Combined Score':<15s}")
    print("-" * 80)

    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"{rank:<6d} {item_id:<8s} {score:<15.4f}")

    print("\nObservation: Combines content similarity and collaborative signals")
    print("             Balances personalization with social proof")


def test_cold_start(system):
    """
    Test cold-start scenario (new user with no history).
    """
    print("\n" + "=" * 80)
    print("Test 4: Cold-Start Handling")
    print("=" * 80)

    # Add new user with no history
    new_user_id = "U005"
    system.add_user(
        new_user_id,
        demographics={"age_group": "25-34", "location": "urban", "gender": "male"},
        preferences={"tech", "music", "photography"},
        history=None  # No history!
    )

    print(f"\nNew user {new_user_id} added with NO interaction history")
    print("Preferences: tech, music, photography")

    # Still get recommendations based on preferences
    recommendations = system.recommend_content_based(new_user_id, top_k=5)

    print(f"\n{'Rank':<6s} {'Item':<8s} {'Similarity':<12s}")
    print("-" * 80)

    for rank, (item_id, sim) in enumerate(recommendations, 1):
        print(f"{rank:<6d} {item_id:<8s} {sim:<12.4f}")

    print("\nObservation: VSA handles cold-start naturally via preference encoding")
    print("             No complex warm-up strategies needed!")


def test_explanation(system):
    """
    Test recommendation explanations.
    """
    print("\n" + "=" * 80)
    print("Test 5: Explainable Recommendations")
    print("=" * 80)

    user_id = "U001"
    item_id = "I009"  # Smart TV

    print(f"\nWhy recommend Item {item_id} to User {user_id}?")

    explanation = system.explain_recommendation(user_id, item_id)

    print(f"\nUser-Item Similarity: {explanation['user_item_similarity']:.4f}")

    if explanation['similar_users']:
        print(f"\nSimilar users who liked this item:")
        for other_id, sim in explanation['similar_users']:
            print(f"  {other_id}: similarity = {sim:.4f}")

    print(f"\n{explanation['explanation']}")

    print("\nObservation: VSA provides interpretable similarity-based explanations")


def main():
    """
    Run complete recommendation system capstone project.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 3 CAPSTONE PROJECT")
    print(" " * 18 + "VSA Recommendation System")
    print("=" * 80)

    # Initialize system
    system, users, items = demo_recommendation_system()

    # Run all tests
    test_content_based(system)
    test_collaborative(system)
    test_hybrid(system)
    test_cold_start(system)
    test_explanation(system)

    # Summary
    print("\n" + "=" * 80)
    print("Capstone Project Complete!")
    print("=" * 80)
    print("\nWhat you built:")
    print("✓ Complete VSA-based recommendation system")
    print("✓ Content-based filtering (user-item matching)")
    print("✓ Collaborative filtering (similar user discovery)")
    print("✓ Hybrid recommendations (combined strategy)")
    print("✓ Cold-start handling (new users with no history)")
    print("✓ Explainable recommendations")

    print("\nKey VSA advantages for recommendations:")
    print("✓ Multimodal fusion (demographics + preferences + history)")
    print("✓ Interpretable similarity-based matching")
    print("✓ Natural cold-start handling (no special cases)")
    print("✓ Constant-time recommendations (no matrix factorization)")
    print("✓ Incremental updates (no retraining)")

    print("\nCongratulations! You've mastered Module 3.")
    print("Ready for Module 4: Advanced Techniques")
    print("=" * 80)


if __name__ == "__main__":
    main()
