# Lesson 3.5: Application - Analogical Reasoning

**Duration:** ~70 minutes (30 min theory + 40 min tutorials)

**Learning Objectives:**

- Understand analogies as transformation mappings
- Create mapping vectors (A:B)
- Apply mappings to solve analogies (A:B::C:?)
- Use Fractional Power Encoding for conceptual spaces
- Complete analogy-solving projects
- Build semantic reasoning systems

---

## Introduction

**Analogies** are at the heart of human reasoning:
- "King is to Queen as Man is to ?"  → Woman
- "Paris is to France as London is to ?" → England
- "Dog is to Puppy as Cat is to ?" → Kitten

VSA naturally supports analogical reasoning through **mapping vectors** - transformations that can be learned and applied.

**What makes VSA good for analogies?**
- ✅ **Algebraic mappings:** Transformations as hypervectors
- ✅ **Compositional:** Can learn and apply transformations
- ✅ **Spatial reasoning:** Conceptual spaces via FPE
- ✅ **Generalizable:** Same mapping applies to different inputs
- ✅ **Interpretable:** Mapping vectors have semantic meaning

---

## Analogies as Transformations

### The Analogy Pattern: A:B::C:?

An analogy states: "A is to B as C is to what?"

**Example:**
```
King : Queen :: Man : ?
```

**Interpretation:** The relationship between King and Queen (male→female royalty) should be the same as the relationship between Man and ? (male→female person).

### Mathematical Formulation

In VSA, we solve analogies by:

1. **Extract transformation:** Learn mapping from A to B
2. **Apply transformation:** Apply same mapping to C
3. **Find result:** Search for most similar concept to transformed C

**Mapping vector:** $$M_{A \to B} = B \otimes A^{-1}$$

**Apply mapping:** $$D = M_{A \to B} \otimes C = (B \otimes A^{-1}) \otimes C$$

**Simplifies to:** $$D \approx \frac{B \otimes C}{A}$$

---

## Creating Mapping Vectors

### Step 1: Encode the Relationship

A **mapping vector** captures the transformation from A to B:

```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add words
memory.add_many(["king", "queen", "man", "woman"])

# Create mapping: king → queen (male royalty to female royalty)
king_vec = memory["king"].vec
queen_vec = memory["queen"].vec

# Mapping = B ⊗ A^(-1)
king_inv = model.opset.inverse(king_vec)
mapping = model.opset.bind(queen_vec, king_inv)

print(f"Mapping: king → queen")
```

**Interpretation:** `mapping` is a hypervector encoding the "make royal male into royal female" transformation.

---

### Step 2: Apply the Mapping

Apply the same transformation to a new word:

```python
# Apply mapping to "man"
man_vec = memory["man"].vec
result = model.opset.bind(mapping, man_vec)

# Find most similar word
from vsax.similarity import cosine_similarity

candidates = ["woman", "queen", "king", "boy", "girl"]
similarities = {c: cosine_similarity(result, memory[c].vec)
                for c in candidates}

answer = max(similarities, key=similarities.get)
print(f"king:queen::man:? → {answer}")  # "woman"
```

**Expected Output:**
```
king:queen::man:? → woman
```

---

## Why Mapping Vectors Work

### Algebra of Transformations

The key insight: **transformations compose algebraically**.

**Derivation:**
```
Mapping M = B ⊗ A^(-1)

Apply to C:
M ⊗ C = (B ⊗ A^(-1)) ⊗ C
      = B ⊗ (A^(-1) ⊗ C)
      ≈ B ⊗ (C ⊗ A^(-1))  (binding is approximately commutative)
```

**Result:** The mapping "rotates" C in hypervector space toward where D should be.

---

### Semantic Similarity of Mappings

**Insight:** Similar transformations create similar mapping vectors!

```python
# Create multiple gender mappings
mappings = {
    "king→queen": model.opset.bind(
        memory["queen"].vec,
        model.opset.inverse(memory["king"].vec)
    ),
    "man→woman": model.opset.bind(
        memory["woman"].vec,
        model.opset.inverse(memory["man"].vec)
    ),
    "prince→princess": model.opset.bind(
        memory["princess"].vec,
        model.opset.inverse(memory["prince"].vec)
    )
}

# Compare mappings
for name1, map1 in mappings.items():
    for name2, map2 in mappings.items():
        if name1 != name2:
            sim = cosine_similarity(map1, map2)
            print(f"Similarity({name1}, {name2}): {sim:.4f}")
```

**Expected:** High similarity (~0.7-0.9) because all represent the same gender transformation!

---

## Types of Analogies

### 1. Semantic Analogies

Relationships between word meanings:

```python
# Examples:
("king", "queen", "man", "woman")           # Gender
("Paris", "France", "London", "England")    # Capital-Country
("dog", "puppy", "cat", "kitten")          # Adult-Young
("big", "bigger", "small", "smaller")      # Comparative
```

---

### 2. Relational Analogies (Kanerva's Framework)

**Kanerva's classic example:** "Dollar of Mexico = ?"

**Encoding:**
```python
# Country-Currency pairs
pairs = [
    ("USA", "dollar"),
    ("Mexico", "peso"),
    ("France", "euro"),
    ("Japan", "yen")
]

# Create mapping: USA → dollar
usa_to_dollar = model.opset.bind(
    memory["dollar"].vec,
    model.opset.inverse(memory["USA"].vec)
)

# Apply to Mexico
result = model.opset.bind(usa_to_dollar, memory["Mexico"].vec)

# Find answer
currencies = ["peso", "dollar", "euro", "yen"]
sims = {c: cosine_similarity(result, memory[c].vec) for c in currencies}
answer = max(sims, key=sims.get)

print(f"Dollar of Mexico = {answer}")  # "peso"
```

---

### 3. Mathematical Analogies

Numeric relationships:

```python
# Successor function: N → N+1
memory.add_many(["one", "two", "three", "four", "five", "six"])

# Learn mapping: one → two
mapping_succ = model.opset.bind(
    memory["two"].vec,
    model.opset.inverse(memory["one"].vec)
)

# Apply to three
result = model.opset.bind(mapping_succ, memory["three"].vec)

numbers = ["one", "two", "three", "four", "five", "six"]
sims = {n: cosine_similarity(result, memory[n].vec) for n in numbers}
answer = max(sims, key=sims.get)

print(f"one:two::three:? → {answer}")  # "four"
```

---

## Conceptual Spaces with FPE

**Fractional Power Encoding** enables analogies in **continuous conceptual spaces**.

### Gärdenfors' Conceptual Spaces

Concepts exist in **multi-dimensional quality spaces**:
- **Colors:** (hue, saturation, brightness)
- **Sounds:** (pitch, loudness, timbre)
- **Emotions:** (valence, arousal)

**FPE** lets us encode these continuous dimensions and reason analogically.

---

### Example: Color Analogies

```python
from vsax.encoders import FractionalPowerEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Define color space dimensions
memory.add_many(["hue", "sat", "bright"])

encoder = FractionalPowerEncoder(model, memory, scale=0.1)

# Encode colors in HSB space
colors = {
    "red": [0, 100, 100],        # Hue=0°, Sat=100%, Bright=100%
    "pink": [350, 50, 100],      # Lighter, less saturated red
    "blue": [240, 100, 100],     # Hue=240°
    "light_blue": [230, 50, 100] # Lighter, less saturated blue
}

color_vecs = {}
for name, [h, s, b] in colors.items():
    vec = encoder.encode_multi(["hue", "sat", "bright"], [h, s, b])
    color_vecs[name] = vec
    memory.add(name)
    memory.symbols[name].vec = vec.vec

# Analogy: red:pink::blue:?
# (Transformation: make lighter and less saturated)

mapping = model.opset.bind(
    color_vecs["pink"].vec,
    model.opset.inverse(color_vecs["red"].vec)
)

result = model.opset.bind(mapping, color_vecs["blue"].vec)

candidates = list(colors.keys())
sims = {c: cosine_similarity(result, color_vecs[c].vec) for c in candidates}
answer = max(sims, key=sims.get)

print(f"red:pink::blue:? → {answer}")  # "light_blue"
```

**Key:** FPE preserves spatial relationships, enabling geometric analogies!

---

## Hands-On: Complete Analogy Tutorials

Now build complete analogy-solving systems!

### Tutorial 3: Kanerva's Classic Analogies

**📓 [Tutorial 3: Kanerva Analogies](../../tutorials/03_kanerva_analogies.md)**

**What you'll learn:**
- Pentti Kanerva's foundational analogy framework
- Country-currency analogies ("Dollar of Mexico")
- Multi-domain analogy solving
- Systematic analogy evaluation

**Time:** ~15 minutes

---

### Tutorial 4: Word Analogies

**📓 [Tutorial 4: Word Analogies](../../tutorials/04_word_analogies.md)**

**What you'll learn:**
- Semantic word analogies (king:queen::man:woman)
- Building analogy test sets
- Accuracy evaluation on standard benchmarks
- Comparing analogy performance across VSA models

**Time:** ~15 minutes

---

### Tutorial 11: Conceptual Spaces & FPE

**📓 [Tutorial 11: Analogical Reasoning with Conceptual Spaces](../../tutorials/11_analogical_reasoning.md)**

**What you'll learn:**
- Gärdenfors' conceptual spaces theory
- Encoding continuous quality dimensions with FPE
- Geometric analogies (color, sound, emotion)
- Spatial transformations in conceptual spaces
- Advanced: Combining symbolic and spatial reasoning

**Time:** ~20 minutes

**Prerequisites:** Understanding of FPE from Lesson 3.1

---

## Key Concepts from the Tutorials

### 1. Mapping Vector Extraction

Learn transformation from examples:

```python
def create_mapping(word_a, word_b, model, memory):
    """Create mapping vector from A to B."""
    a_inv = model.opset.inverse(memory[word_a].vec)
    mapping = model.opset.bind(memory[word_b].vec, a_inv)
    return mapping
```

---

### 2. Mapping Application

Apply learned transformation to new input:

```python
def apply_mapping(word, mapping, model, memory):
    """Apply mapping to word."""
    result = model.opset.bind(memory[word].vec, mapping)
    return result
```

---

### 3. Cleanup and Search

Find best match to result:

```python
def solve_analogy(a, b, c, candidates, model, memory):
    """Solve A:B::C:?"""
    # Create mapping
    mapping = create_mapping(a, b, model, memory)

    # Apply to C
    result = apply_mapping(c, mapping, model, memory)

    # Find best match
    sims = {cand: cosine_similarity(result, memory[cand].vec)
            for cand in candidates}

    return max(sims, key=sims.get)
```

---

### 4. Conceptual Space Encoding

Multi-dimensional quality spaces with FPE:

```python
# Encode concept in 3D quality space
concept_hv = encoder.encode_multi(
    ["dimension1", "dimension2", "dimension3"],
    [value1, value2, value3]
)
```

---

### 5. Geometric Transformations

Spatial analogies preserve geometric relationships:

```python
# Transformation: shift in hue space
# red (hue=0) → blue (hue=240)
# orange (hue=30) → ? (should be hue≈270, violet)
```

---

## Extensions and Experiments

### 1. Cross-Domain Analogies

Analogies spanning multiple domains:

```python
# Musical analogy: "Do:Re::Red:?"
# (Musical note progression ≈ Color spectrum progression)

# Encode both domains with shared "progression" mapping
```

---

### 2. Proportional Analogies

Encode proportional relationships:

```python
# "Hand is to Arm as Foot is to ?" → Leg
# (Part-whole relationship)

memory.add_many(["hand", "arm", "foot", "leg", "finger", "toe"])

# Create part→whole mapping
part_whole = create_mapping("hand", "arm", model, memory)

# Apply
result = apply_mapping("foot", part_whole, model, memory)
# Should retrieve "leg"
```

---

### 3. Temporal Analogies

Encode sequential patterns:

```python
# "Morning is to Breakfast as Evening is to ?" → Dinner
# (Time-of-day to meal mapping)
```

---

### 4. Hierarchical Analogies

Multi-level transformations:

```python
# "Species:Genus::Genus:?" → Family
# (Taxonomic hierarchy)
```

---

## Real-World Applications

VSA analogies are used in:

**1. Natural Language Understanding**
- Word sense disambiguation
- Metaphor interpretation
- Semantic similarity tasks

**2. Educational Systems**
- Automated analogy generation for tests
- Student reasoning assessment
- Adaptive learning (present analogies at right difficulty)

**3. Creative AI**
- Metaphor generation
- Conceptual blending
- Design by analogy (transfer solutions across domains)

**4. Scientific Discovery**
- Cross-domain knowledge transfer
- Hypothesis generation via analogy
- "What is the X of domain Y?"

**5. Case-Based Reasoning**
- Legal reasoning: "This case is like Case X in that..."
- Medical diagnosis: "Symptoms A:B::Symptoms C:?"

---

## Comparison: VSA vs Word Embeddings (Word2Vec)

| Feature | VSA Analogies | Word2Vec Analogies |
|---------|---------------|-------------------|
| **Method** | Explicit mapping vectors (B⊗A^(-1)) | Vector arithmetic (B - A + C) |
| **Training** | No training (compositional) | Requires large corpus |
| **Interpretability** | Mapping = explicit transformation | Embedding dimensions opaque |
| **Continuous spaces** | FPE for conceptual spaces | Not designed for continuous |
| **Compositionality** | Fully compositional | Approximately compositional |
| **Few-shot** | Works with 1-2 examples | Needs large corpus |

**VSA advantages:**
- ✅ No training corpus needed
- ✅ Fully compositional and algebraic
- ✅ Explicit semantic transformations
- ✅ Conceptual spaces with FPE

**Word2Vec advantages:**
- ✅ Higher accuracy on standard benchmarks (trained on large corpora)
- ✅ Captures statistical co-occurrence

---

## Self-Assessment

Before moving to the next module, ensure you can:

- [ ] Create mapping vectors (M = B ⊗ A^(-1))
- [ ] Apply mappings to solve analogies
- [ ] Understand why similar transformations have similar mappings
- [ ] Use FPE for conceptual space analogies
- [ ] Complete all three analogy tutorials
- [ ] Build custom analogy solvers
- [ ] Evaluate analogy accuracy
- [ ] Extend to multi-domain and geometric analogies

---

## Quick Quiz

**Q1:** How is a mapping vector M from word A to word B created?

a) M = A ⊗ B
b) M = A ⊕ B
c) M = B ⊗ A^(-1)
d) M = B - A

<details>
<summary>Answer</summary>
**c) M = B ⊗ A^(-1)** - The mapping is created by binding B with the inverse of A. This creates a transformation vector that, when applied to A, retrieves B.
</details>

**Q2:** To solve the analogy "A:B::C:?", what is the complete process?

a) Create mapping M = B⊗A^(-1), apply to C: D = M⊗C, find closest match
b) Compute C - A + B
c) Bundle A, B, C and search
d) Bind A⊗B⊗C

<details>
<summary>Answer</summary>
**a) Create mapping, apply, search** - First extract the transformation from A to B, then apply that same transformation to C to get a result vector D, finally search for the concept most similar to D.
</details>

**Q3:** Why does FPE enable geometric analogies in conceptual spaces?

a) FPE is faster than regular encoding
b) FPE preserves spatial relationships between continuous values
c) FPE works with all VSA models
d) FPE requires no training

<details>
<summary>Answer</summary>
**b) FPE preserves spatial relationships** - FPE uses phase rotation (v^r = exp(i*r*θ)) which maintains geometric structure. Nearby values in input space → nearby hypervectors, enabling analogies like "red:pink::blue:light_blue" (same lightening transformation).
</details>

**Q4:** What does it mean if two mapping vectors have high similarity?

a) The source words are similar
b) The target words are similar
c) The transformations are semantically similar
d) The mappings are incorrect

<details>
<summary>Answer</summary>
**c) The transformations are similar** - High similarity between mapping vectors means they represent similar transformations. For example, "king→queen" and "man→woman" both represent the gender transformation, so their mapping vectors will be similar.
</details>

---

## Key Takeaways

1. **Analogies as transformations** - A:B :: C:? means "apply A→B transformation to C"
2. **Mapping vectors** - M = B ⊗ A^(-1) encodes the transformation
3. **Algebraic composition** - Transformations compose: M ⊗ C = D
4. **Similar mappings** - Same transformation across different word pairs → similar mapping vectors
5. **Conceptual spaces** - FPE enables geometric analogies in continuous quality spaces
6. **Fully compositional** - No training needed, works with few examples
7. **Interpretable** - Mapping vectors have explicit semantic meaning

---

**Next:** [Module 4: Advanced Techniques](../04_advanced/index.md)

Dive into advanced VSA capabilities: Clifford operators, spatial semantic pointers, hierarchical structures, and multi-modal fusion.

**Previous:** [Lesson 3.4: Application - Knowledge Graph Reasoning](04_knowledge_graphs.md)

---

## Module 3 Complete!

🎉 **Congratulations!** You've completed Module 3: Encoders & Applications.

**You've learned:**
- ✅ ScalarEncoder and FractionalPowerEncoder for continuous data
- ✅ SequenceEncoder for ordered data
- ✅ DictEncoder, SetEncoder, GraphEncoder for structured data
- ✅ Image classification with prototype learning
- ✅ Knowledge graph reasoning and querying
- ✅ Analogical reasoning and conceptual spaces

**Skills acquired:**
- Encode any data type as hypervectors
- Build real-world VSA applications
- Query knowledge bases compositionally
- Solve analogies algebraically
- Combine multiple encoders for complex tasks

**Ready for advanced techniques in Module 4!**
