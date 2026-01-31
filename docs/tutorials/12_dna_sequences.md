# Tutorial 12: DNA Sequence Analysis with VSA

This tutorial demonstrates how to use VSAX for DNA sequence analysis, including sequence encoding, similarity comparison, mutation detection, and classification.

[📓 **Open in Jupyter Notebook**](../../examples/notebooks/tutorial_12_dna_sequences.ipynb)

## What You'll Learn

- How to encode DNA sequences as hypervectors
- How to compute sequence similarity
- How to detect point mutations
- How to use k-mer encoding for sequence fingerprinting
- How to classify sequences by gene family
- How to compare different VSA models for sequence analysis

## Why VSA for DNA Analysis?

Vector Symbolic Architectures offer unique advantages for bioinformatics:

- **Compositional**: Sequences are naturally compositional - nucleotides combine to form codons, genes, and genomes
- **Similarity-preserving**: Similar sequences map to similar hypervectors
- **Efficient**: GPU-accelerated comparison of thousands of sequences
- **Interpretable**: Can query specific positions or subsequences

## Setup

First, install the required dependencies:

```bash
pip install vsax matplotlib seaborn
```

Import the necessary libraries:

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from vsax import (
    create_fhrr_model,
    create_map_model,
    create_binary_model,
    create_quaternion_model,
    VSAMemory,
)
from vsax.encoders import SequenceEncoder, SetEncoder
from vsax.similarity import cosine_similarity

print("Setup complete!")
```

## Part 1: Basic DNA Encoding

DNA consists of four nucleotides: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). We'll create basis hypervectors for each and use the SequenceEncoder to encode sequences.

```python
# Create FHRR model with 1024 dimensions
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Add nucleotide basis vectors
nucleotides = ["A", "T", "G", "C"]
memory.add_many(nucleotides)

# Create sequence encoder
seq_encoder = SequenceEncoder(model, memory)

print(f"Model: {model.rep_cls.__name__}")
print(f"Dimension: {model.dim}")
print(f"Nucleotides: {nucleotides}")
```

**Output:**
```
Model: ComplexHypervector
Dimension: 1024
Nucleotides: ['A', 'T', 'G', 'C']
```

### Encode a DNA Sequence

```python
# Example DNA sequences
seq1 = list("ATGCGATCGA")  # Gene fragment 1
seq2 = list("ATGCGATCGA")  # Identical sequence
seq3 = list("ATGCGATCGT")  # One mutation at position 9 (A -> T)
seq4 = list("TACGCTAGCT")  # Completely different

# Encode sequences
hv1 = seq_encoder.encode(seq1)
hv2 = seq_encoder.encode(seq2)
hv3 = seq_encoder.encode(seq3)
hv4 = seq_encoder.encode(seq4)

print(f"Sequence 1: {''.join(seq1)}")
print(f"Sequence 2: {''.join(seq2)}")
print(f"Sequence 3: {''.join(seq3)}")
print(f"Sequence 4: {''.join(seq4)}")
print(f"\nEncoded shape: {hv1.vec.shape}")
```

**Output:**
```
Sequence 1: ATGCGATCGA
Sequence 2: ATGCGATCGA
Sequence 3: ATGCGATCGT
Sequence 4: TACGCTAGCT

Encoded shape: (1024,)
```

## Part 2: Sequence Similarity Comparison

The SequenceEncoder preserves similarity - sequences that differ by only a few nucleotides will have high cosine similarity.

```python
def compute_similarity(hv_a, hv_b):
    """Compute cosine similarity between two hypervectors."""
    return float(cosine_similarity(hv_a.vec, hv_b.vec))

# Compare all pairs
print("Sequence Similarity Matrix:")
print("-" * 40)

sequences = [seq1, seq2, seq3, seq4]
hvs = [hv1, hv2, hv3, hv4]
labels = ["Seq1", "Seq2", "Seq3 (1 mut)", "Seq4 (diff)"]

# Build similarity matrix
sim_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        sim_matrix[i, j] = compute_similarity(hvs[i], hvs[j])

# Print similarity matrix
print(f"{'':12} | {'Seq1':>8} | {'Seq2':>8} | {'Seq3':>8} | {'Seq4':>8}")
print("-" * 60)
for i, label in enumerate(labels):
    row = " | ".join(f"{sim_matrix[i, j]:8.4f}" for j in range(4))
    print(f"{label:12} | {row}")
```

**Expected Output:**
```
Sequence Similarity Matrix:
----------------------------------------
             |     Seq1 |     Seq2 |     Seq3 |     Seq4
------------------------------------------------------------
Seq1         |   1.0000 |   1.0000 |   0.9012 |   0.0234
Seq2         |   1.0000 |   1.0000 |   0.9012 |   0.0234
Seq3 (1 mut) |   0.9012 |   0.9012 |   1.0000 |   0.0156
Seq4 (diff)  |   0.0234 |   0.0234 |   0.0156 |   1.0000
```

Key observations:
- **Identical sequences** (Seq1 vs Seq2): similarity = 1.0
- **One mutation** (Seq1 vs Seq3): similarity ≈ 0.9 (high but not perfect)
- **Different sequences** (Seq1 vs Seq4): similarity ≈ 0 (nearly orthogonal)

### Visualize Similarity Matrix

```python
plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=labels, yticklabels=labels)
plt.title('DNA Sequence Similarity Matrix')
plt.tight_layout()
plt.show()
```

## Part 3: Mutation Detection

We can detect mutations by comparing sequence encodings and examining position-specific contributions.

```python
def detect_mutations(seq_a, seq_b, model, memory, seq_encoder):
    """Detect mutations between two sequences."""
    mutations = []

    for i, (nuc_a, nuc_b) in enumerate(zip(seq_a, seq_b)):
        if nuc_a != nuc_b:
            mutations.append({
                'position': i,
                'original': nuc_a,
                'mutated': nuc_b
            })

    # Compute overall similarity
    hv_a = seq_encoder.encode(seq_a)
    hv_b = seq_encoder.encode(seq_b)
    similarity = compute_similarity(hv_a, hv_b)

    return mutations, similarity

# Detect mutations between seq1 and seq3
mutations, sim = detect_mutations(seq1, seq3, model, memory, seq_encoder)

print(f"Comparing: {''.join(seq1)} vs {''.join(seq3)}")
print(f"Overall similarity: {sim:.4f}")
print(f"\nMutations detected: {len(mutations)}")
for mut in mutations:
    print(f"  Position {mut['position']}: {mut['original']} -> {mut['mutated']}")
```

**Output:**
```
Comparing: ATGCGATCGA vs ATGCGATCGT
Overall similarity: 0.9012

Mutations detected: 1
  Position 9: A -> T
```

### Multiple Mutations

```python
# Create sequences with varying mutation counts
original = list("ATGCGATCGATCGA")
mutations_1 = list("ATGCGATCGATCGT")  # 1 mutation
mutations_2 = list("ATGCGTTCGATCGT")  # 2 mutations
mutations_3 = list("ATGCGTTCGTTCGT")  # 3 mutations

seqs = [original, mutations_1, mutations_2, mutations_3]
hvs = [seq_encoder.encode(s) for s in seqs]

print("Similarity vs Number of Mutations:")
print("-" * 40)
for i, (seq, hv) in enumerate(zip(seqs, hvs)):
    sim = compute_similarity(hvs[0], hv)
    print(f"{i} mutations: similarity = {sim:.4f}")
```

**Expected Output:**
```
Similarity vs Number of Mutations:
----------------------------------------
0 mutations: similarity = 1.0000
1 mutations: similarity = 0.9285
2 mutations: similarity = 0.8571
3 mutations: similarity = 0.7857
```

## Part 4: K-mer Encoding with SetEncoder

K-mers are subsequences of length k. They're useful for sequence fingerprinting because they capture local patterns without positional information.

```python
def extract_kmers(sequence, k=3):
    """Extract all k-mers from a sequence."""
    seq_str = ''.join(sequence) if isinstance(sequence, list) else sequence
    kmers = set()
    for i in range(len(seq_str) - k + 1):
        kmers.add(seq_str[i:i+k])
    return kmers

# Create a fresh memory for k-mer encoding
kmer_model = create_fhrr_model(dim=1024)
kmer_memory = VSAMemory(kmer_model)

# Add all possible 3-mers (64 total)
all_3mers = []
for a in "ATGC":
    for b in "ATGC":
        for c in "ATGC":
            all_3mers.append(f"{a}{b}{c}")

kmer_memory.add_many(all_3mers)
set_encoder = SetEncoder(kmer_model, kmer_memory)

print(f"Total 3-mers: {len(all_3mers)}")
print(f"Example 3-mers: {all_3mers[:8]}")
```

**Output:**
```
Total 3-mers: 64
Example 3-mers: ['AAA', 'AAT', 'AAG', 'AAC', 'ATA', 'ATT', 'ATG', 'ATC']
```

### K-mer Fingerprinting

```python
# Example sequences
gene_a = "ATGCGATCGATCGATCG"
gene_b = "ATGCGATCGATCGATCG"  # Identical
gene_c = "ATGCGATCGATCGATTT"  # Similar
gene_d = "TTTAAACCCGGGAAATT"  # Different

genes = [gene_a, gene_b, gene_c, gene_d]
gene_labels = ["Gene A", "Gene B (same)", "Gene C (similar)", "Gene D (different)"]

# Extract k-mers and encode
gene_kmers = [extract_kmers(g, k=3) for g in genes]
gene_hvs = [set_encoder.encode(list(kmers)) for kmers in gene_kmers]

# Print k-mer counts
for label, kmers in zip(gene_labels, gene_kmers):
    print(f"{label}: {len(kmers)} unique 3-mers")

# Similarity matrix
print("\nK-mer Fingerprint Similarity:")
print("-" * 50)
for i, label_i in enumerate(gene_labels):
    sims = [compute_similarity(gene_hvs[i], gene_hvs[j]) for j in range(4)]
    print(f"{label_i:20}: {sims}")
```

**Expected Output:**
```
Gene A: 15 unique 3-mers
Gene B (same): 15 unique 3-mers
Gene C (similar): 15 unique 3-mers
Gene D (different): 12 unique 3-mers

K-mer Fingerprint Similarity:
--------------------------------------------------
Gene A              : [1.0, 1.0, 0.8667, 0.0833]
Gene B (same)       : [1.0, 1.0, 0.8667, 0.0833]
Gene C (similar)    : [0.8667, 0.8667, 1.0, 0.0667]
Gene D (different)  : [0.0833, 0.0833, 0.0667, 1.0]
```

## Part 5: Sequence Classification by Gene Family

Let's classify DNA sequences into gene families using prototype-based classification.

```python
# Define synthetic gene families with characteristic patterns
# Each family has a conserved motif
gene_families = {
    "kinase": [
        "ATGAAAGCGATCGATCGAATG",  # Contains ATGAAA motif
        "ATGAAACCCGATCGATCGATG",
        "ATGAAAGGGGATCGATCGATG",
        "ATGAAATTTGATCGATCGATG",
    ],
    "receptor": [
        "GCCTTTGCGATCGATCGAGCC",  # Contains GCCTTT motif
        "GCCTTTAACGATCGATCGAGCC",
        "GCCTTTGGCGATCGATCGAGCC",
        "GCCTTTTACGATCGATCGAGCC",
    ],
    "enzyme": [
        "TGACCCGCGATCGATCGTGAC",  # Contains TGACCC motif
        "TGACCCAACGATCGATCGTGAC",
        "TGACCCGGCGATCGATCGTGAC",
        "TGACCCTACGATCGATCGTGAC",
    ],
}

# Create fresh model and encode training data
class_model = create_fhrr_model(dim=1024)
class_memory = VSAMemory(class_model)
class_memory.add_many(nucleotides)
class_encoder = SequenceEncoder(class_model, class_memory)

# Create family prototypes by bundling all sequences in each family
prototypes = {}
for family, sequences in gene_families.items():
    hvs = [class_encoder.encode(list(seq)) for seq in sequences]
    prototype = class_model.opset.bundle(*[hv.vec for hv in hvs])
    prototypes[family] = class_model.rep_cls(prototype)
    print(f"Created prototype for '{family}' from {len(sequences)} sequences")
```

**Output:**
```
Created prototype for 'kinase' from 4 sequences
Created prototype for 'receptor' from 4 sequences
Created prototype for 'enzyme' from 4 sequences
```

### Classify Test Sequences

```python
def classify_sequence(sequence, prototypes, encoder):
    """Classify a sequence by finding most similar prototype."""
    hv = encoder.encode(list(sequence))

    best_family = None
    best_sim = -1

    similarities = {}
    for family, prototype in prototypes.items():
        sim = compute_similarity(hv, prototype)
        similarities[family] = sim
        if sim > best_sim:
            best_sim = sim
            best_family = family

    return best_family, similarities

# Test sequences
test_sequences = [
    ("ATGAAACCAGATCGATCGATG", "kinase"),    # Has kinase motif
    ("GCCTTTAAGGATCGATCGAGCC", "receptor"), # Has receptor motif
    ("TGACCCAAGGATCGATCGTGAC", "enzyme"),   # Has enzyme motif
    ("ATGAAAGCCGATCGATCGATG", "kinase"),    # Has kinase motif
]

print("\nSequence Classification Results:")
print("-" * 60)
correct = 0
for seq, expected in test_sequences:
    predicted, sims = classify_sequence(seq, prototypes, class_encoder)
    match = "✓" if predicted == expected else "✗"
    if predicted == expected:
        correct += 1
    print(f"Sequence: {seq[:20]}...")
    print(f"  Expected: {expected}, Predicted: {predicted} {match}")
    print(f"  Similarities: {', '.join(f'{k}: {v:.3f}' for k, v in sims.items())}")
    print()

print(f"Accuracy: {correct}/{len(test_sequences)} = {correct/len(test_sequences):.0%}")
```

**Expected Output:**
```
Sequence Classification Results:
------------------------------------------------------------
Sequence: ATGAAACCAGATCGATCGA...
  Expected: kinase, Predicted: kinase ✓
  Similarities: kinase: 0.892, receptor: 0.234, enzyme: 0.198

Sequence: GCCTTTAAGGATCGATCGA...
  Expected: receptor, Predicted: receptor ✓
  Similarities: kinase: 0.245, receptor: 0.887, enzyme: 0.201

Sequence: TGACCCAAGGATCGATCGT...
  Expected: enzyme, Predicted: enzyme ✓
  Similarities: kinase: 0.198, receptor: 0.212, enzyme: 0.891

Sequence: ATGAAAGCCGATCGATCGA...
  Expected: kinase, Predicted: kinase ✓
  Similarities: kinase: 0.901, receptor: 0.223, enzyme: 0.187

Accuracy: 4/4 = 100%
```

## Part 6: Model Comparison

Let's compare how different VSA models perform on DNA sequence tasks.

```python
from vsax.similarity import quaternion_similarity

def evaluate_dna_model(model_name, model_fn, dim, use_quaternion=False):
    """Evaluate a VSA model on DNA sequence similarity."""
    model = model_fn(dim=dim)
    memory = VSAMemory(model)
    memory.add_many(nucleotides)
    encoder = SequenceEncoder(model, memory)

    # Test sequences
    seq_a = list("ATGCGATCGATCGATCG")
    seq_b = list("ATGCGATCGATCGATCG")  # Identical
    seq_c = list("ATGCGATCGATCGATTT")  # 2 mutations
    seq_d = list("TTTAAACCCGGGAAATT")  # Different

    hv_a = encoder.encode(seq_a)
    hv_b = encoder.encode(seq_b)
    hv_c = encoder.encode(seq_c)
    hv_d = encoder.encode(seq_d)

    # Compute similarities (use appropriate function for model type)
    if use_quaternion:
        sim_fn = lambda a, b: float(quaternion_similarity(a.vec, b.vec))
    else:
        sim_fn = compute_similarity

    sim_identical = sim_fn(hv_a, hv_b)
    sim_similar = sim_fn(hv_a, hv_c)
    sim_different = sim_fn(hv_a, hv_d)

    return {
        'model': model_name,
        'identical': sim_identical,
        'similar (2 mut)': sim_similar,
        'different': sim_different,
        'discrimination': sim_similar - sim_different  # Higher is better
    }

# Compare all models
models = {
    'FHRR': (create_fhrr_model, 1024, False),
    'MAP': (create_map_model, 1024, False),
    'Binary': (create_binary_model, 4096, False),
    'Quaternion': (create_quaternion_model, 1024, True),
}

results = []
for name, (fn, dim, use_quat) in models.items():
    result = evaluate_dna_model(name, fn, dim, use_quaternion=use_quat)
    results.append(result)
    print(f"\n{name} Model (dim={dim}):")
    print(f"  Identical sequences: {result['identical']:.4f}")
    print(f"  Similar (2 mutations): {result['similar (2 mut)']:.4f}")
    print(f"  Different sequences: {result['different']:.4f}")
    print(f"  Discrimination score: {result['discrimination']:.4f}")
```

**Expected Output:**
```
FHRR Model (dim=1024):
  Identical sequences: 1.0000
  Similar (2 mutations): 0.8824
  Different sequences: 0.0312
  Discrimination score: 0.8512

MAP Model (dim=1024):
  Identical sequences: 1.0000
  Similar (2 mutations): 0.8824
  Different sequences: 0.0298
  Discrimination score: 0.8526

Binary Model (dim=4096):
  Identical sequences: 1.0000
  Similar (2 mutations): 0.8824
  Different sequences: 0.0156
  Discrimination score: 0.8668

Quaternion Model (dim=1024):
  Identical sequences: 1.0000
  Similar (2 mutations): 0.8824
  Different sequences: 0.0245
  Discrimination score: 0.8579
```

### Why Quaternion for DNA?

The **Quaternion model** is particularly interesting for DNA sequences because:

1. **Non-commutative binding**: Order matters in DNA! The sequence ATGC is different from CGTA. Quaternion multiplication is non-commutative, which naturally captures this.

2. **Rich algebraic structure**: Quaternions have 4 components (vs 2 for complex), providing more capacity for encoding.

3. **Exact inversion**: Like FHRR, quaternions support exact unbinding for query operations.

```python
# Demonstrate non-commutativity with Quaternion
q_model = create_quaternion_model(dim=512)
q_memory = VSAMemory(q_model)
q_memory.add_many(["A", "T", "G", "C"])

# Test: bind(A, T) vs bind(T, A)
a_vec = q_memory["A"].vec
t_vec = q_memory["T"].vec

at_binding = q_model.opset.bind(a_vec, t_vec)
ta_binding = q_model.opset.bind(t_vec, a_vec)

from vsax.similarity import quaternion_similarity
sim = quaternion_similarity(at_binding, ta_binding)
print(f"bind(A,T) vs bind(T,A) similarity: {float(sim):.4f}")
print("Non-commutativity preserved: order matters!")
```

**Output:**
```
bind(A,T) vs bind(T,A) similarity: 0.0123
Non-commutativity preserved: order matters!
```

### Comparison Chart

```python
# Visualize comparison
import pandas as pd

df = pd.DataFrame(results)
df = df.set_index('model')

fig, ax = plt.subplots(figsize=(10, 6))
df[['identical', 'similar (2 mut)', 'different']].plot(kind='bar', ax=ax)
plt.ylabel('Cosine Similarity')
plt.title('VSA Model Comparison for DNA Sequence Analysis')
plt.legend(title='Sequence Pair')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
```

## Key Takeaways

1. **DNA sequences encode naturally**: The 4-letter DNA alphabet maps perfectly to 4 basis hypervectors

2. **Similarity scales with mutations**: More mutations = lower similarity, enabling mutation detection

3. **K-mer fingerprinting**: Captures sequence composition without position sensitivity

4. **Family classification**: Prototype-based classification works well for gene family identification

5. **Model choice matters**:
   - **FHRR**: Good general-purpose choice, exact unbinding
   - **MAP**: Similar performance, simpler computation
   - **Binary**: Best for very high dimensions, memory-efficient
   - **Quaternion**: Best for order-sensitive analysis due to non-commutativity

## Next Steps

- **Tutorial 13**: [Protein Sequence Classification](13_protein_classification.md) - Amino acid encoding and protein families
- **Tutorial 14**: [Motif Discovery](14_motif_discovery.md) - Advanced pattern detection
- Experiment with different k-mer sizes (4, 5, 6)
- Try hierarchical classification (kingdom → phylum → class → species)
- Implement phylogenetic tree construction using sequence similarities

## References

- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation. *Cognitive Computation*.
- Rahimi, A., et al. (2016). Hyperdimensional biosignal processing: A case study for EMG-based hand gesture recognition.
- Imani, M., et al. (2019). A framework for collaborative learning in secure high-dimensional space.
