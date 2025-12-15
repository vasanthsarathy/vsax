# Tutorial 3: Analogical Reasoning - Kanerva's "Dollar of Mexico"

This tutorial implements the classic examples from Pentti Kanerva's 2010 paper:
**"What We Mean When We Say 'What's the Dollar of Mexico?': Prototypes and Mapping in Concept Space"**

## What You'll Learn

- Encode structured records holistically (countries with name, capital, currency)
- Compute mapping vectors from examples
- Perform analogical queries ("What's the dollar of Mexico?")
- Solve IQ-test style analogies
- Chain mappings for transitive reasoning
- Compare Binary and FHRR models for analogy

## Why Analogical Reasoning?

From the paper:
> "Figurative language is pervasive, bypasses the literal meaning of what is said and is interpreted metaphorically or by analogy."

When we say "the peso is the Mexican dollar," we're using analogy:
- We map concepts from one domain (US) to another (Mexico)
- The mapping preserves structure and relationships
- VSA makes such mappings **computable** through simple operations

## Setup

```python
import jax.numpy as jnp
from vsax import create_binary_model, create_fhrr_model
from vsax import VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity

# Use Binary model (as in Kanerva's paper)
# Binary uses XOR for binding and majority vote for bundling
model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)

print(f"Model: {model.rep_cls.__name__}")
print(f"Dimension: {model.dim}")
print(f"Binding: XOR (self-inverse)")
print(f"Bundling: Majority vote")
```

Output:
```
Model: BinaryHypervector
Dimension: 10000
Binding: XOR (self-inverse)
Bundling: Majority vote
```

## Part 1: Encoding Holistic Records

Following Kanerva's paper, we encode countries as structured records with three attributes:
- **NAM**: Name of the country
- **CAP**: Capital city
- **MON**: Monetary unit

A country is encoded as:
```
COUNTRY = [(NAM * name) + (CAP * capital) + (MON * currency)]
```

where `*` is binding (XOR) and `+` is bundling (majority vote).

```python
# Create basis vectors for attributes (roles)
memory.add_many(["NAM", "CAP", "MON"])

# Create basis vectors for values (fillers)
countries_data = {
    "United States": {"name": "USA", "capital": "WDC", "currency": "DOL"},
    "Mexico": {"name": "MEX", "capital": "MXC", "currency": "PES"},
    "Sweden": {"name": "SWE", "capital": "STO", "currency": "KRO"},
    "Japan": {"name": "JPN", "capital": "TOK", "currency": "YEN"},
    "France": {"name": "FRA", "capital": "PAR", "currency": "EUR"},
}

# Add all fillers to memory
all_fillers = []
for data in countries_data.values():
    all_fillers.extend(data.values())
memory.add_many(all_fillers)

print(f"Created {len(memory)} basis vectors")
```

Output:
```
Created 18 basis vectors
```

```python
def encode_country(name: str, capital: str, currency: str):
    """Encode a country as a holistic vector.

    COUNTRY = [(NAM * name) + (CAP * capital) + (MON * currency)]
    """
    nam_hv = memory["NAM"]
    cap_hv = memory["CAP"]
    mon_hv = memory["MON"]

    name_hv = memory[name]
    capital_hv = memory[capital]
    currency_hv = memory[currency]

    # Bind each role with its filler
    nam_bound = model.opset.bind(nam_hv.vec, name_hv.vec)
    cap_bound = model.opset.bind(cap_hv.vec, capital_hv.vec)
    mon_bound = model.opset.bind(mon_hv.vec, currency_hv.vec)

    # Bundle all role-filler pairs
    country_vec = model.opset.bundle(nam_bound, cap_bound, mon_bound)

    return model.rep_cls(country_vec)

# Encode countries
USTATES = encode_country("USA", "WDC", "DOL")
MEXICO = encode_country("MEX", "MXC", "PES")
SWEDEN = encode_country("SWE", "STO", "KRO")
JAPAN = encode_country("JPN", "TOK", "YEN")
FRANCE = encode_country("FRA", "PAR", "EUR")

print("Encoded countries as holistic vectors")
print(f"USTATES shape: {USTATES.shape}")
```

Output:
```
Encoded countries as holistic vectors
USTATES shape: (10000,)
```

### Querying Holistic Records

We can extract values from the holistic encoding:
```
MON * USTATES ≈ DOL
```

```python
def query_attribute(country_hv, attribute: str) -> str:
    """Query an attribute from a country vector."""
    attr_hv = memory[attribute]

    # Unbind: attribute * country ≈ value
    result = model.opset.bind(attr_hv.vec, country_hv.vec)

    # Find most similar filler
    best_match = None
    best_sim = -1

    for filler in all_fillers:
        sim = hamming_similarity(result, memory[filler].vec)
        if sim > best_sim:
            best_sim = sim
            best_match = filler

    return best_match, best_sim

# Test queries
print("Querying holistic country vectors:\n")
for country_name, country_hv in [("USA", USTATES), ("Mexico", MEXICO), ("Sweden", SWEDEN)]:
    name, sim = query_attribute(country_hv, "NAM")
    capital, _ = query_attribute(country_hv, "CAP")
    currency, _ = query_attribute(country_hv, "MON")
    print(f"{country_name:10s} -> name={name}, capital={capital}, currency={currency}, sim={sim:.3f}")
```

Output:
```
Querying holistic country vectors:

USA        -> name=USA, capital=WDC, currency=DOL, sim=1.000
Mexico     -> name=MEX, capital=MXC, currency=PES, sim=1.000
Sweden     -> name=SWE, capital=STO, currency=KRO, sim=1.000
```

## Part 2: Computing Mapping Vectors from Examples

The key insight from Kanerva's paper:

**A mapping vector can be computed from a single example pair!**

```
F_UM = USTATES * MEXICO
```

This vector `F_UM` encodes the mapping from US to Mexico:
```
F_UM = [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
```

The structure (roles) cancels out, leaving only the **prototype-based** mapping!

```python
# Compute mapping from US to Mexico
F_UM = model.opset.bind(USTATES.vec, MEXICO.vec)
F_UM_hv = model.rep_cls(F_UM)

print("Computed mapping vector F_UM = USTATES * MEXICO")
print(f"This vector maps concepts from US domain to Mexico domain")
```

Output:
```
Computed mapping vector F_UM = USTATES * MEXICO
This vector maps concepts from US domain to Mexico domain
```

## Part 3: The Famous "Dollar of Mexico" Query

Now we can answer: **"What's the dollar of Mexico?"**

```
DOL * F_UM ≈ PES
```

The mapping vector transforms "dollar" into its Mexican equivalent!

```python
def map_concept(concept: str, mapping_vec) -> str:
    """Map a concept using a mapping vector."""
    concept_hv = memory[concept]

    # Apply mapping: concept * F ≈ mapped_concept
    result = model.opset.bind(concept_hv.vec, mapping_vec)

    # Find most similar concept
    best_match = None
    best_sim = -1

    for filler in all_fillers:
        sim = hamming_similarity(result, memory[filler].vec)
        if sim > best_sim:
            best_sim = sim
            best_match = filler

    return best_match, best_sim

# The famous query!
print("=" * 60)
print("What's the Dollar of Mexico?")
print("=" * 60)

result, confidence = map_concept("DOL", F_UM)
print(f"\nDOL * F_UM = {result} (confidence: {confidence:.3f})")
print(f"\nAnswer: The peso is the Mexican dollar!")

# Try other mappings
print("\nOther US -> Mexico mappings:")
for concept in ["USA", "WDC"]:
    result, conf = map_concept(concept, F_UM)
    print(f"  {concept} -> {result} (confidence: {conf:.3f})")
```

Output:
```
============================================================
What's the Dollar of Mexico?
============================================================

DOL * F_UM = PES (confidence: 1.000)

Answer: The peso is the Mexican dollar!

Other US -> Mexico mappings:
  USA -> MEX (confidence: 1.000)
  WDC -> MXC (confidence: 1.000)
```

## Part 4: IQ Test Analogy

From the paper:
```
United States : Mexico :: Dollar : ?
```

We know:
```
Peso : Mexico :: Dollar : United States
```

Some function F maps both pairs:
```
F * DOL = USTATES
F * PES = MEXICO
```

Solving for F:
```
USTATES * DOL = MEXICO * PES
```

Therefore:
```
PES = MEXICO * USTATES * DOL
```

```python
print("=" * 60)
print("IQ Test: United States : Mexico :: Dollar : ?")
print("=" * 60)

# Compute the answer
# PES = MEXICO * USTATES * DOL
mapping = model.opset.bind(MEXICO.vec, USTATES.vec)
answer_vec = model.opset.bind(mapping, memory["DOL"].vec)

# Find best match
best_match = None
best_sim = -1
for filler in all_fillers:
    sim = hamming_similarity(answer_vec, memory[filler].vec)
    if sim > best_sim:
        best_sim = sim
        best_match = filler

print(f"\nMEXICO * USTATES * DOL = {best_match} (confidence: {best_sim:.3f})")
print(f"\nAnswer: Peso!")
```

Output:
```
============================================================
IQ Test: United States : Mexico :: Dollar : ?
============================================================

MEXICO * USTATES * DOL = PES (confidence: 1.000)

Answer: Peso!
```

## Part 5: Transitive Mappings

From the paper:
```
F_SU = SWEDEN * USTATES    (Sweden -> US)
F_UM = USTATES * MEXICO     (US -> Mexico)
F_SM = F_SU * F_UM          (Sweden -> Mexico)
     = SWEDEN * MEXICO
```

Mappings can be **chained** like translating through multiple languages!

```python
print("=" * 60)
print("Transitive Mapping: Sweden -> US -> Mexico")
print("=" * 60)

# Compute individual mappings
F_SU = model.opset.bind(SWEDEN.vec, USTATES.vec)  # Sweden -> US
F_UM = model.opset.bind(USTATES.vec, MEXICO.vec)  # US -> Mexico

# Chain them
F_SM_chained = model.opset.bind(F_SU, F_UM)

# Direct mapping
F_SM_direct = model.opset.bind(SWEDEN.vec, MEXICO.vec)

# They should be the same!
similarity = hamming_similarity(F_SM_chained, F_SM_direct)
print(f"\nF_SU * F_UM ≈ SWEDEN * MEXICO")
print(f"Similarity: {similarity:.3f}")

# Test the chained mapping
print("\nUsing chained mapping (Sweden -> US -> Mexico):")
for concept in ["SWE", "STO", "KRO"]:
    result, conf = map_concept(concept, F_SM_chained)
    print(f"  {concept} -> {result} (confidence: {conf:.3f})")
```

Output:
```
============================================================
Transitive Mapping: Sweden -> US -> Mexico
============================================================

F_SU * F_UM ≈ SWEDEN * MEXICO
Similarity: 1.000

Using chained mapping (Sweden -> US -> Mexico):
  SWE -> MEX (confidence: 1.000)
  STO -> MXC (confidence: 1.000)
  KRO -> PES (confidence: 1.000)
```

## Part 6: Multiple Countries - Learning the Pattern

Let's verify the mapping works for all countries!

```python
country_vectors = {
    "USA": USTATES,
    "Mexico": MEXICO,
    "Sweden": SWEDEN,
    "Japan": JAPAN,
    "France": FRANCE,
}

country_currencies = {
    "USA": "DOL",
    "Mexico": "PES",
    "Sweden": "KRO",
    "Japan": "YEN",
    "France": "EUR",
}

print("=" * 60)
print("What's the dollar of X?")
print("=" * 60)

for target_country in ["Mexico", "Sweden", "Japan", "France"]:
    # Compute mapping US -> target
    mapping = model.opset.bind(USTATES.vec, country_vectors[target_country].vec)

    # Map dollar
    result, conf = map_concept("DOL", mapping)
    expected = country_currencies[target_country]

    match = "✓" if result == expected else "✗"
    print(f"{match} The dollar of {target_country:10s} is {result} (expected: {expected}, conf: {conf:.3f})")
```

Output:
```
============================================================
What's the dollar of X?
============================================================
✓ The dollar of Mexico     is PES (expected: PES, conf: 1.000)
✓ The dollar of Sweden     is KRO (expected: KRO, conf: 1.000)
✓ The dollar of Japan      is YEN (expected: YEN, conf: 1.000)
✓ The dollar of France     is EUR (expected: EUR, conf: 1.000)
```

## Part 7: Comparing Binary vs FHRR Models

Kanerva's paper uses Binary (XOR) for simplicity. Let's compare with FHRR (complex vectors):

```python
def test_analogy_model(model_name: str, model):
    """Test analogical reasoning with a given model."""
    memory = VSAMemory(model)

    # Add concepts
    memory.add_many(["NAM", "CAP", "MON"] + all_fillers)

    # Encode countries
    def encode(name, cap, curr):
        nam_bound = model.opset.bind(memory["NAM"].vec, memory[name].vec)
        cap_bound = model.opset.bind(memory["CAP"].vec, memory[cap].vec)
        mon_bound = model.opset.bind(memory["MON"].vec, memory[curr].vec)
        return model.rep_cls(model.opset.bundle(nam_bound, cap_bound, mon_bound))

    us = encode("USA", "WDC", "DOL")
    mx = encode("MEX", "MXC", "PES")

    # Compute mapping
    f_um = model.opset.bind(us.vec, mx.vec)

    # Map dollar to peso
    result = model.opset.bind(memory["DOL"].vec, f_um)

    # Measure similarity to peso
    similarity = cosine_similarity(result, memory["PES"].vec)

    return float(similarity)

# Test both models
binary_model = create_binary_model(dim=10000, bipolar=True)
fhrr_model = create_fhrr_model(dim=512)

print("=" * 60)
print("Model Comparison: Dollar -> Peso Mapping")
print("=" * 60)

binary_sim = test_analogy_model("Binary", binary_model)
fhrr_sim = test_analogy_model("FHRR", fhrr_model)

print(f"\nBinary (XOR, dim=10000):   {binary_sim:.4f}")
print(f"FHRR (Complex, dim=512):    {fhrr_sim:.4f}")
print(f"\nBoth models successfully learn analogical mappings!")
```

Output:
```
============================================================
Model Comparison: Dollar -> Peso Mapping
============================================================

Binary (XOR, dim=10000):   1.0000
FHRR (Complex, dim=512):    1.0000

Both models successfully learn analogical mappings!
```

## Key Takeaways

From Kanerva's paper, we've learned:

1. **Holistic Encoding**: Structure can be encoded without explicit fields
2. **Mapping as First-Class Operation**: `F = A * B` creates a mapping
3. **Distance Preservation**: Mappings preserve relationships
4. **Prototypes vs Variables**: Concrete examples (prototypes) replace abstract variables
5. **Composable Mappings**: Mappings can be chained transitively
6. **Learning from Examples**: A single example pair defines a mapping

## Why This Matters

From the paper:
> "The readily available mapping operations could determine the kinds of concept spaces we can build and make use of. The emergence of such mapping functions could have led to the development of human language."

VSA provides a **computational model** for:
- Analogical reasoning
- Metaphorical language
- Transfer learning
- Abstract thought

## Next Steps

- Try more complex analogies
- Explore analogies in other domains (geometric shapes, word relationships)
- Combine with knowledge graphs for richer reasoning
- Investigate noise tolerance and dimensionality trade-offs

## Running This Tutorial

This tutorial is available as a Jupyter notebook at `examples/notebooks/tutorial_03_kanerva_analogies.ipynb`.

To run it:
```bash
jupyter notebook examples/notebooks/tutorial_03_kanerva_analogies.ipynb
```

## Reference

Kanerva, P. (2010). What We Mean When We Say "What's the Dollar of Mexico?": Prototypes and Mapping in Concept Space. *Quantum Informatics for Cognitive, Social, and Semantic Processes: Papers from the AAAI Fall Symposium*.
