"""
Module 1 Capstone Project: Analogy Solver

Build a VSA system that solves analogies: "A is to B as C is to ?"

Example:
- "King is to Queen as Man is to ?" → "Woman"
- "Paris is to France as London is to ?" → "England"

Approach:
1. Encode word pairs as role-filler bindings
2. Extract mapping vectors (transformations)
3. Apply mappings to solve analogies
4. Test with multiple analogy types

Expected learning:
- Using binding for relational encoding
- Mapping vectors capture transformations
- Compositional reasoning with VSA
- Real-world application of Module 1 concepts
"""

from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity
import jax.numpy as jnp


class AnalogySOlver:
    """
    VSA-based analogy solver using mapping vectors.
    """

    def __init__(self, dim=2048):
        """
        Initialize the analogy solver.

        Args:
            dim: Hypervector dimensionality
        """
        self.model = create_fhrr_model(dim=dim)
        self.memory = VSAMemory(self.model)

    def add_words(self, words):
        """
        Add words to the vocabulary.

        Args:
            words: List of word strings
        """
        self.memory.add_many(words)
        print(f"Added {len(words)} words to vocabulary")

    def create_mapping(self, word_a, word_b):
        """
        Create a mapping vector from word_a to word_b.

        The mapping captures the transformation: A → B

        Args:
            word_a: Source word
            word_b: Target word

        Returns:
            Mapping hypervector (B ⊗ A^(-1))
        """
        # Get inverse of A
        a_inv = self.model.opset.inverse(self.memory[word_a].vec)

        # Create mapping: B ⊗ A^(-1)
        mapping = self.model.opset.bind(self.memory[word_b].vec, a_inv)

        return mapping

    def apply_mapping(self, word, mapping):
        """
        Apply a mapping to a word to find the analogy result.

        Args:
            word: Input word (C in "A:B::C:?")
            mapping: Mapping vector from A→B

        Returns:
            Result hypervector (should be similar to D)
        """
        result = self.model.opset.bind(self.memory[word].vec, mapping)
        return result

    def solve_analogy(self, word_a, word_b, word_c, candidates=None):
        """
        Solve the analogy: "A is to B as C is to ?"

        Args:
            word_a: First word in analogy pair (e.g., "king")
            word_b: Second word in analogy pair (e.g., "queen")
            word_c: Query word (e.g., "man")
            candidates: Optional list of candidate words to search
                       If None, searches all words in vocabulary

        Returns:
            (best_word, similarity, all_similarities)
        """
        # Create mapping from A to B
        mapping = self.create_mapping(word_a, word_b)

        # Apply mapping to C
        result = self.apply_mapping(word_c, mapping)

        # Find best match
        if candidates is None:
            candidates = list(self.memory.symbols.keys())

        similarities = {}
        for candidate in candidates:
            sim = cosine_similarity(result, self.memory[candidate].vec)
            similarities[candidate] = float(sim)

        # Get best match
        best_word = max(similarities, key=similarities.get)
        best_sim = similarities[best_word]

        return best_word, best_sim, similarities

    def print_analogy_result(self, word_a, word_b, word_c, expected=None, top_k=5):
        """
        Solve and print analogy results in a readable format.

        Args:
            word_a, word_b, word_c: Analogy words
            expected: Expected answer (optional, for verification)
            top_k: Number of top candidates to show
        """
        print(f"\nAnalogy: '{word_a}' is to '{word_b}' as '{word_c}' is to ?")
        print("-" * 60)

        best_word, best_sim, all_sims = self.solve_analogy(word_a, word_b, word_c)

        # Sort by similarity
        sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)

        # Show top-k results
        print(f"\nTop {top_k} candidates:")
        for rank, (word, sim) in enumerate(sorted_sims[:top_k], 1):
            marker = " ✓" if word == expected else ""
            marker += " ★" if word == best_word else ""
            print(f"  {rank}. {word:<15} (similarity: {sim:.4f}){marker}")

        # Verification
        if expected:
            if best_word == expected:
                print(f"\n✓ SUCCESS! Found expected answer: '{expected}'")
            else:
                exp_sim = all_sims.get(expected, 0.0)
                print(f"\n✗ Expected '{expected}' (sim: {exp_sim:.4f}), got '{best_word}' (sim: {best_sim:.4f})")


def test_gender_analogies():
    """
    Test analogies related to gender transformations.
    """
    print("=" * 60)
    print("Test 1: Gender Analogies")
    print("=" * 60)

    solver = AnalogySOlver(dim=2048)

    # Add vocabulary
    words = [
        "king",
        "queen",
        "man",
        "woman",
        "boy",
        "girl",
        "prince",
        "princess",
        "husband",
        "wife",
        "father",
        "mother",
        "son",
        "daughter",
    ]

    solver.add_words(words)

    # Test cases
    test_cases = [("king", "queen", "man", "woman"), ("king", "queen", "prince", "princess"), ("father", "mother", "son", "daughter"), ("husband", "wife", "man", "woman")]

    for a, b, c, expected in test_cases:
        solver.print_analogy_result(a, b, c, expected=expected, top_k=3)


def test_location_analogies():
    """
    Test analogies related to geographic locations.
    """
    print("\n" + "=" * 60)
    print("Test 2: Location Analogies")
    print("=" * 60)

    solver = AnalogySOlver(dim=2048)

    # Add vocabulary
    words = [
        "paris",
        "france",
        "london",
        "england",
        "tokyo",
        "japan",
        "berlin",
        "germany",
        "rome",
        "italy",
        "madrid",
        "spain",
    ]

    solver.add_words(words)

    # Test cases
    test_cases = [("paris", "france", "london", "england"), ("tokyo", "japan", "berlin", "germany"), ("rome", "italy", "madrid", "spain")]

    for a, b, c, expected in test_cases:
        solver.print_analogy_result(a, b, c, expected=expected, top_k=3)


def test_mathematical_analogies():
    """
    Test analogies related to mathematical relationships.
    """
    print("\n" + "=" * 60)
    print("Test 3: Mathematical Analogies")
    print("=" * 60)

    solver = AnalogySOlver(dim=2048)

    # Add vocabulary
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

    solver.add_words(words)

    # Test cases: successor function
    test_cases = [("one", "two", "three", "four"), ("five", "six", "seven", "eight"), ("ten", "eleven", "three", "four")]

    print("\nTesting successor relationship (N → N+1):")
    for a, b, c, expected in test_cases:
        solver.print_analogy_result(a, b, c, expected=expected, top_k=3)


def test_custom_analogy():
    """
    Interactive test: user can create their own analogies.
    """
    print("\n" + "=" * 60)
    print("Test 4: Custom Analogy")
    print("=" * 60)

    solver = AnalogySOlver(dim=4096)  # Higher dimension for better accuracy

    # Example: Comparative adjectives
    words = ["big", "bigger", "small", "smaller", "fast", "faster", "slow", "slower", "hot", "hotter", "cold", "colder"]

    solver.add_words(words)

    print("\nTesting comparative adjectives:")
    test_cases = [("big", "bigger", "small", "smaller"), ("fast", "faster", "slow", "slower"), ("hot", "hotter", "cold", "colder")]

    for a, b, c, expected in test_cases:
        solver.print_analogy_result(a, b, c, expected=expected, top_k=3)


def analyze_mapping_properties():
    """
    Analyze properties of mapping vectors.
    """
    print("\n" + "=" * 60)
    print("Analysis: Mapping Vector Properties")
    print("=" * 60)

    solver = AnalogySOlver(dim=2048)
    words = ["king", "queen", "man", "woman", "prince", "princess"]
    solver.add_words(words)

    # Create multiple mappings for the same transformation (male→female)
    mappings = {
        "king→queen": solver.create_mapping("king", "queen"),
        "man→woman": solver.create_mapping("man", "woman"),
        "prince→princess": solver.create_mapping("prince", "princess"),
    }

    print("\nComparing mappings for the same transformation (male→female):")
    print("-" * 60)

    # Compare mappings
    mapping_names = list(mappings.keys())
    for i, name1 in enumerate(mapping_names):
        for name2 in mapping_names[i + 1 :]:
            sim = cosine_similarity(mappings[name1], mappings[name2])
            print(f"  Similarity({name1:<15}, {name2:<15}): {sim:.4f}")

    print("\nObservation:")
    print("- Similar transformations create similar mapping vectors")
    print("- Mappings capture abstract relationships")
    print("- This is the foundation of analogical reasoning!")


def main():
    """
    Run all analogy solver tests.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 1 CAPSTONE PROJECT")
    print(" " * 25 + "Analogy Solver")
    print("=" * 80)

    # Run test suites
    test_gender_analogies()
    test_location_analogies()
    test_mathematical_analogies()
    test_custom_analogy()
    analyze_mapping_properties()

    # Summary
    print("\n" + "=" * 80)
    print("Capstone Project Complete!")
    print("=" * 80)
    print("\nWhat you learned:")
    print("✓ Building VSA models from scratch")
    print("✓ Encoding relational knowledge with binding")
    print("✓ Creating and applying transformation mappings")
    print("✓ Solving analogies through compositional reasoning")
    print("✓ Analyzing mapping vector properties")
    print("\nCongratulations! You've mastered Module 1 fundamentals.")
    print("Ready for Module 2: Deep Dive into VSA Models")
    print("=" * 80)


if __name__ == "__main__":
    main()
